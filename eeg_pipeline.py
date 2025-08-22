import os
import math
import argparse
import random
from dataclasses import dataclass
from typing import Tuple, List, Optional
from dataclasses import asdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Preprocessing & IO
import mne
import pandas as pd

# Wavelet / Fourier
import pywt
from scipy import signal

# Metrics
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

SEED = 1337
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

def set_deterministic():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# -----------------------------
# Data utilities
# -----------------------------
@dataclass
class PreprocCfg:
    l_freq: float = 0.5
    h_freq: float = 45.0
    notch_hz: float = 50.0  # change to 60.0 if required
    resample_hz: Optional[int] = 200
    reref: str = "average"  # or "cz", "fpz", etc
    epoch_sec: float = 4.0
    epoch_overlap: float = 0.5  # 50% overlap
    channels: Optional[List[str]] = None  # pick commonly available, e.g., ["Fz","Cz","Pz","Oz","F3","F4","C3","C4","P3","P4","O1","O2"]

@dataclass
class FeatureCfg:
    feature: str = "cwt"  # "cwt" or "stft" or "raw"
    # CWT params
    wavelet: str = "morl"
    cwt_scales: int = 64
    # STFT params
    stft_nperseg: int = 256
    stft_noverlap: int = 128

class EEGDataset(Dataset):
    def __init__(self, labels_csv: str, split: str, split_col: str = "split",
                 pre: PreprocCfg = PreprocCfg(), feat: FeatureCfg = FeatureCfg()):
        super().__init__()
        self.df = pd.read_csv(labels_csv)
        # Expect columns: subject_id, filepath, label, split  (split in {train,val,test})
        assert {"subject_id", "filepath", "label"}.issubset(self.df.columns), "labels.csv must have subject_id, filepath, label"
        if split_col in self.df.columns:
            self.df = self.df[self.df[split_col] == split].reset_index(drop=True)
        self.pre = pre
        self.feat = feat

        # Precompute channel picks if provided
        self.pick_channels = self.pre.channels

    def __len__(self):
        return len(self.df)

    def _load_eeg(self, fpath: str) -> mne.io.BaseRaw:
        ext = os.path.splitext(fpath)[1].lower()
        if ext in [".edf", ".bdf"]:
            raw = mne.io.read_raw_edf(fpath, preload=True, verbose="ERROR")
        elif ext in [".fif"]:
            raw = mne.io.read_raw_fif(fpath, preload=True, verbose="ERROR")
        else:
            raise ValueError(f"Unsupported EEG file: {fpath}")
        return raw

    def _preprocess(self, raw: mne.io.BaseRaw) -> np.ndarray:
        # Pick channels if specified
        if self.pick_channels is not None:
            picks = mne.pick_channels(raw.info["ch_names"], include=self.pick_channels)
            raw.pick(picks)
        # Notch & bandpass
        raw.notch_filter(self.pre.notch_hz, verbose="ERROR")
        raw.filter(self.pre.l_freq, self.pre.h_freq, verbose="ERROR")
        # Re-reference
        if self.pre.reref == "average":
            raw.set_eeg_reference("average", verbose="ERROR")
        elif self.pre.reref in raw.info["ch_names"]:
            raw.set_eeg_reference([self.pre.reref], verbose="ERROR")
        # Resample
        if self.pre.resample_hz is not None:
            raw.resample(self.pre.resample_hz)
        data = raw.get_data()  # shape: (C, T)
        return data

    def _window(self, data: np.ndarray, sfreq: float) -> List[np.ndarray]:
        win = int(self.pre.epoch_sec * sfreq)
        step = int(win * (1 - self.pre.epoch_overlap))
        windows = []
        for start in range(0, data.shape[1] - win + 1, step):
            windows.append(data[:, start:start+win])
        return windows

    def _make_cwt(self, x: np.ndarray, sfreq: float) -> np.ndarray:
        # x: (C, T)
        # Return: image-like tensor (CWT stacked along channels) shape (1 or C, scales, time)
        scales = np.linspace(1, self.feat.cwt_scales, num=self.feat.cwt_scales)
        imgs = []
        for ch in x:
            coef, _ = pywt.cwt(ch, scales, self.feat.wavelet, sampling_period=1.0/sfreq)
            # coef: (scales, T)
            imgs.append(np.abs(coef).astype(np.float32))
        img = np.stack(imgs, axis=0)  # (C, scales, T)
        return img

    def _make_stft(self, x: np.ndarray, sfreq: float) -> np.ndarray:
        # Compute magnitude spectrogram per channel and stack
        fts = []
        for ch in x:
            f, t, Zxx = signal.stft(ch, fs=sfreq, nperseg=self.feat.stft_nperseg, noverlap=self.feat.stft_noverlap)
            fts.append(np.abs(Zxx).astype(np.float32))  # (F, T')
        ft = np.stack(fts, axis=0)  # (C, F, T')
        return ft

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        raw = self._load_eeg(row["filepath"])  # preload=True
        sfreq = raw.info["sfreq"]
        data = self._preprocess(raw)
        windows = self._window(data, sfreq)
        label = int(row["label"])  # 0 or 1

        # Choose a random window per _getitem_
        x = random.choice(windows)
        if self.feat.feature == "cwt":
            feat = self._make_cwt(x, sfreq)  # (C, scales, T)
        elif self.feat.feature == "stft":
            feat = self._make_stft(x, sfreq)  # (C, F, T')
        elif self.feat.feature == "raw":
            feat = x.astype(np.float32)  # (C,T)
        else:
            raise ValueError("Unknown feature type")

        # Normalize per-sample
        feat = feat - feat.mean()
        std = feat.std() + 1e-6
        feat = feat / std

        # Return tensor and label
        return torch.tensor(feat, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

# -----------------------------
# Models
# -----------------------------
class CNN2D(nn.Module):
    def __init__(self, in_ch: int, n_classes: int = 2):
        super().__init__()
        # in_ch = number of EEG channels (we treat each channel as an input channel)
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.AdaptiveAvgPool2d((8,8))
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*8*8, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, n_classes)
        )

    def forward(self, x):
        # x: (B, C, H, W) where C=EEG channels, H=scales or freqs, W=time
        z = self.net(x)
        return self.head(z)

class Transformer1D(nn.Module):
    def __init__(self, in_ch: int, seq_len: int, d_model: int = 128, nhead: int = 4, num_layers: int = 4, n_classes: int = 2):
        super().__init__()
        # Flatten channel × features over freq into tokens along time
        self.proj = nn.Linear(in_ch, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.enc = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.cls = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, n_classes)
        )
        self.pos = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.01)

    def forward(self, x):
        # For STFT we expect x shape: (B, C, F, T). We'll average over F to get (B, T, C)
        if x.dim() == 4:
            x = x.mean(dim=2)  # (B, C, T)
            x = x.permute(0, 2, 1)  # (B, T, C)
        elif x.dim() == 3:  # raw: (B, C, T)
            x = x.permute(0, 2, 1)
        else:
            raise ValueError("Unexpected input shape for Transformer1D")
        B, T, C = x.shape
        z = self.proj(x) + self.pos[:, :T, :]
        z = self.enc(z)
        # CLS via mean pooling
        z = z.mean(dim=1)
        return self.cls(z)

# -----------------------------
# Training / Evaluation
# -----------------------------

def make_loaders(labels_csv: str, batch_size: int, pre: PreprocCfg, feat: FeatureCfg):
    # If no split column is present, we will create subject-wise splits here
    df = pd.read_csv(labels_csv)
    if "split" not in df.columns:
        # subject-wise split 70/15/15
        subjects = df["subject_id"].unique().tolist()
        random.shuffle(subjects)
        n = len(subjects)
        n_train = int(0.7*n)
        n_val = int(0.15*n)
        train_subj = set(subjects[:n_train])
        val_subj = set(subjects[n_train:n_train+n_val])
        df["split"] = df["subject_id"].apply(lambda s: "train" if s in train_subj else ("val" if s in val_subj else "test"))
        tmp = os.path.join(os.path.dirname(labels_csv), "labels.with_splits.csv")
        df.to_csv(tmp, index=False)
        labels_csv = tmp

    ds_tr = EEGDataset(labels_csv, split="train", pre=pre, feat=feat)
    ds_va = EEGDataset(labels_csv, split="val", pre=pre, feat=feat)
    ds_te = EEGDataset(labels_csv, split="test", pre=pre, feat=feat)

    def _infer_in_ch(ds: EEGDataset):
        # sample one to get channels & spatial dims
        x, y = ds[0]
        return x.shape[0], x.shape[1:], y

    in_ch, spatial, _ = _infer_in_ch(ds_tr)

    dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)
    dl_va = DataLoader(ds_va, batch_size=batch_size, shuffle=False, num_workers=0)
    dl_te = DataLoader(ds_te, batch_size=batch_size, shuffle=False, num_workers=0)

    return dl_tr, dl_va, dl_te, in_ch, spatial


def train_one_epoch(model, loader, opt, device):
    model.train()
    total, correct = 0, 0
    losses = []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        # Reshape for CNN/Transformer expectations
        if isinstance(model, CNN2D):
            # x: (B, C, H, W) OK for CWT/STFT
            if x.dim() == 3:  # raw (B,C,T) → fake HxW: (B,C,1,T)
                x = x.unsqueeze(2)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return np.mean(losses), correct/total

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    y_true, y_prob, y_pred = [], [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if isinstance(model, CNN2D):
            if x.dim() == 3:
                x = x.unsqueeze(2)
        logits = model(x)
        prob = F.softmax(logits, dim=1)[:,1]
        y_true.extend(y.cpu().numpy().tolist())
        y_prob.extend(prob.cpu().numpy().tolist())
        y_pred.extend(logits.argmax(dim=1).cpu().numpy().tolist())
    try:
        auroc = roc_auc_score(y_true, y_prob)
    except Exception:
        auroc = float('nan')
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    return {"auroc": auroc, "acc": acc, "f1": f1}

# -----------------------------
# Main
# -----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, required=True)
    parser.add_argument('--labels', type=str, default='labels.csv', help='CSV with subject_id,filepath,label[,split]')
    parser.add_argument('--feature', type=str, default='cwt', choices=['cwt','stft','raw'])
    parser.add_argument('--model', type=str, default='cnn2d', choices=['cnn2d','transformer1d'])
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--save_dir', type=str, default='checkpoints')
    parser.add_argument('--channels', type=str, default='', help='Comma list to pick specific channels, empty = all available')
    parser.add_argument('--reref', type=str, default='average')
    parser.add_argument('--epoch_sec', type=float, default=4.0)
    parser.add_argument('--overlap', type=float, default=0.5)
    parser.add_argument('--notch', type=float, default=50.0)
    parser.add_argument('--band', type=str, default='0.5-45')
    parser.add_argument('--resample', type=int, default=200)

    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    low, high = map(float, args.band.split('-'))
    channels = [c.strip() for c in args.channels.split(',') if c.strip()] or None

    pre = PreprocCfg(
        l_freq=low, h_freq=high, notch_hz=args.notch, resample_hz=args.resample,
        reref=args.reref, epoch_sec=args.epoch_sec, epoch_overlap=args.overlap,
        channels=channels
    )
    feat = FeatureCfg(feature=args.feature)

    labels_csv = os.path.join(args.dataset_dir, args.labels)

    dl_tr, dl_va, dl_te, in_ch, spatial = make_loaders(labels_csv, args.batch_size, pre, feat)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.model == 'cnn2d':
        model = CNN2D(in_ch=in_ch, n_classes=2).to(device)
    else:
        # Infer seq_len from spatial dims (for STFT/raw)
        if args.feature == 'stft':
            # spatial = (F, T)
            seq_len = spatial[1]
        elif args.feature == 'raw':
            # spatial = (T,)
            seq_len = spatial[0]
        else:  # cwt → treat like image; transformer less meaningful here
            seq_len = spatial[-1]
        model = Transformer1D(in_ch=in_ch, seq_len=seq_len, n_classes=2).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_val = -1
    for epoch in range(1, args.epochs+1):
        tr_loss, tr_acc = train_one_epoch(model, dl_tr, opt, device)
        val_metrics = evaluate(model, dl_va, device)
        score = val_metrics['auroc'] if not math.isnan(val_metrics['auroc']) else val_metrics['acc']
        print(f"Epoch {epoch:02d} | loss {tr_loss:.4f} | tr_acc {tr_acc:.3f} | val_acc {val_metrics['acc']:.3f} | val_f1 {val_metrics['f1']:.3f} | val_auroc {val_metrics['auroc']:.3f}")
        if score > best_val:
            best_val = score
            ckpt_path = os.path.join(args.save_dir, f"best_{args.model}_{args.feature}.pt")
            torch.save({
                'model_state_dict': model.state_dict(),
                'preprocessing': asdict(pre),
                'features': asdict(feat),
            }, ckpt_path)
            print(f"Saved checkpoint → {ckpt_path}")

    # Final test
    test_metrics = evaluate(model, dl_te, device)
    print(f"TEST | acc {test_metrics['acc']:.3f} | f1 {test_metrics['f1']:.3f} | auroc {test_metrics['auroc']:.3f}")

if __name__ == "__main__":
    main()
