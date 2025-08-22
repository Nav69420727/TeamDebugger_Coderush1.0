import mne
import numpy as np
import pandas as pd
from pathlib import Path
import shutil

# === CONFIGURATION ===
source_file = Path("C:/Users/DELL/Documents/VSC/Schizophrenia Detecion using EEG and deep learning/ML/eeg_data.fif")
output_dir = Path("C:/Users/DELL/Documents/VSC/Schizophrenia Detecion using EEG and deep learning/ML/synthetic_data")
n_samples_per_class = 10  # how many synthetic files per class

# Create output directory
output_dir.mkdir(exist_ok=True)

# Load the original EEG file
raw = mne.io.read_raw_fif(source_file, preload=True)

# === FUNCTION TO CREATE VARIATION ===
def create_variant(raw, noise_level=1e-6, scale_range=(0.95, 1.05)):
    data = raw.get_data()
    noise = np.random.randn(*data.shape) * noise_level
    scale = np.random.uniform(*scale_range)
    new_data = data * scale + noise

    new_raw = raw.copy()
    new_raw._data = new_data
    return new_raw

# === GENERATE SYNTHETIC DATA ===
records = []

for cls in ["Control", "Schizophrenia"]:
    for i in range(n_samples_per_class):
        variant = create_variant(raw)
        out_file = output_dir / f"{cls.lower()}_{i+1}.fif"
        variant.save(out_file, overwrite=True)
        records.append({"subject": out_file.name, "label": cls})

# === SAVE LABELS CSV ===
labels_df = pd.DataFrame(records)
labels_csv_path = output_dir / "labels.csv"
labels_df.to_csv(labels_csv_path, index=False)

print(f"Synthetic dataset created at: {output_dir}")
print(f"Labels file: {labels_csv_path}")
