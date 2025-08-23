
import matplotlib.pyplot as plt
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import LeaveOneGroupOut

import mne
from mne.datasets.sleep_physionet.age import fetch_data
from features import eeg_power_band, FREQ_BANDS

# Experiment variables

# classification label target
EVENT_ID = {
    "Sleep stage W": 1,
    "Sleep stage 1/2/3/4": 2,
    "Sleep stage R": 3,
}

# Frequency bands range in Hz for EEG

# Number of participants used in the analysis
NUM_PARTICIPANT = 5

"""# Step 1: Loading the Data and Preprocessing"""

def load_data(participant_id, event_id=EVENT_ID):
  """ Will load the EDF with annotation for a given participant and create
      30 seconds epochs

  Parameters
  ----------
  participant_id:
    The subjects to use. Can be in the range of 0-82 (inclusive), however the
    following subjects are not available: 39, 68, 69, 78 and 79.


  Return
  ----------
  raw_edf:
    Contains the edf with the annotations
  events:
    Contains the 30 seconds events
  epochs:
    the 30 seconds epoch

  Limitation:
  -----------
    Will only get 1 recording session
    Will only work for 1 subject at a time
  """

  ANNOTATION_EVENT_ID = {
    "Sleep stage W": 1,
    "Sleep stage 1": 2,
    "Sleep stage 2": 2,
    "Sleep stage 3": 2,
    "Sleep stage 4": 2,
    "Sleep stage R": 3,
  }

  # Load two file paths, one for the signal and one for the annotations
  [participant_file] = fetch_data(subjects=[participant_id], recording=[1])

  # Read the signal file with information on each signal
  raw_edf = mne.io.read_raw_edf(
    participant_file[0],
    stim_channel="Event marker",
    infer_types=True,
    preload=True,
    verbose="error"
  )

  # Read the annotation file
  annotation_edf = mne.read_annotations(participant_file[1])

  # keep last 4h wake events before sleep and first 4h wake events after
  # sleep and redefine annotations on raw data
  annotation_edf.crop(annotation_edf[1]["onset"] - 30 * 240, annotation_edf[-2]["onset"] + 30 * 240)

  # Attach the annotation file to the raw edf loaded
  raw_edf.set_annotations(annotation_edf, emit_warning=False)

  # Chunk the data into 30 seconds epochs
  events, _ = mne.events_from_annotations(
      raw_edf, event_id=ANNOTATION_EVENT_ID, chunk_duration=30.0
  )


  # Create the epochs so that we can use it for classification
  tmax = 30.0 - 1.0 / raw_edf.info["sfreq"]  # tmax in included

  epochs = mne.Epochs(
      raw=raw_edf,
      events=events,
      event_id=event_id,
      tmin=0.0,
      tmax=tmax,
      baseline=None,
      preload=True,
  )

  return raw_edf, events, epochs

# Load NUM_PARTICIPANT data and store them for further processing
all_participant_epochs = []
for participant_id in range(NUM_PARTICIPANT):
  _, _, epochs = load_data(participant_id=participant_id)
  all_participant_epochs.append(epochs)

all_participant_epochs

"""# Step 2: Feature Calculation"""

def eeg_power_band(epochs, freq_bands=FREQ_BANDS):
  """Calculate relative spectral analysis on the EEG sensors for each epochs"""

  # Calculate the spectrogram
  spectrum = epochs.compute_psd(picks="eeg", fmin=0.5, fmax=30.0)
  psds, freqs = spectrum.get_data(return_freqs=True)

  # Normalization
  psds /= np.sum(psds, axis=-1, keepdims=True)

  # shape of PSDS:
  # (epoch, number of channels (we have two), frequency_bins)
  # We'll slice and average to get the delta to theta bands (5 feature per channel)
  # Therefore we should finish with (epoch, number of channels * number of bands)

  X = []
  # For each frequency band get the mean value and add it to the list X
  for fmin, fmax in freq_bands.values():
    psds_band = psds[:, :, (freqs >= fmin) & (freqs < fmax)].mean(axis=-1)
    X.append(psds_band.reshape(len(psds), -1))

  # return a numpy array, by reshuffling the list from a (5,841,2) to a (841,10)
  return np.concatenate(X, axis=1)

# Iterate over the 30 seconds epochs, calculate the proper powerband features
# Then define the right id for the "group" which should be the participant_id

X = []
y = []
groups = []
for group_id, epochs in enumerate(all_participant_epochs):
  print(f"Processing participants #{group_id}")

  X_epoch = eeg_power_band(epochs)
  y_epoch = epochs.events[:, 2]
  group_epoch = [group_id]*len(y_epoch)

  X.append(X_epoch)
  y.append(y_epoch)
  groups.append(group_epoch)

# Transform these lists into numpy array with proper size for sklearn models
X = np.concatenate(X, axis=0)
y = np.concatenate(y, axis=0)
groups = np.concatenate(groups, axis=0)

# Leave One Participant Out Cross Validation, using group_id == participant_id
logo = LeaveOneGroupOut()

group_id = 0
for train, test in logo.split(X, y, groups=groups):
  print(f"Testing on participant: #{group_id}")
  group_id = group_id + 1

  # Training of the classifier
  model = RandomForestClassifier(n_estimators=100, random_state=42)
  X_train = X[train]
  y_train = y[train]
  model.fit(X_train, y_train)

  # Testing using the current participant data left out
  X_test = X[test]
  y_test = y[test]
  y_pred = model.predict(X_test)

  acc = accuracy_score(y_test, y_pred)

  print(f"Accuracy score: {acc}")

  # Create a confusion matrix and a report on all the metrics
  cm = confusion_matrix(y_test, y_pred)
  disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
  disp.plot() # will appear at the end of the output

  print(classification_report(y_test, y_pred, target_names=EVENT_ID.keys()))
  print(EVENT_ID)

"""## Analysis Steps for Two Participants with One Recording for Binary State

"""

# Experiment variables

# classification label target
EVENT_ID = {
    "Awake": 1,
    "Sleeping": 2,
}


"""**bold text**# Step 1: Loading the Data and Preprocessing"""

def load_data(participant_id, event_id=EVENT_ID):
  """ Will load the EDF with annotation for a given participant and create
      30 seconds epochs

  Parameters
  ----------
  participant_id:
    The subjects to use. Can be in the range of 0-82 (inclusive), however the
    following subjects are not available: 39, 68, 69, 78 and 79.


  Return
  ----------
  raw_edf:
    Contains the edf with the annotations
  events:
    Contains the 30 seconds events
  epochs:
    the 30 seconds epoch

  Limitation:
  -----------
    Will only get 1 recording session
    Will only work for 1 subject at a time
  """

  ANNOTATION_EVENT_ID = {
    "Sleep stage W": 1,
    "Sleep stage 1": 2,
    "Sleep stage 2": 2,
    "Sleep stage 3": 2,
    "Sleep stage 4": 2,
    "Sleep stage R": 2,
  }

  # Load two file paths, one for the signal and one for the annotations
  [participant_file] = fetch_data(subjects=[participant_id], recording=[1])

  # Read the signal file with information on each signal
  raw_edf = mne.io.read_raw_edf(
    participant_file[0],
    stim_channel="Event marker",
    infer_types=True,
    preload=True,
    verbose="error"
  )

  # Read the annotation file
  annotation_edf = mne.read_annotations(participant_file[1])

  # keep last 4h wake events before sleep and first 4h wake events after
  # sleep and redefine annotations on raw data
  annotation_edf.crop(annotation_edf[1]["onset"] - 30 * 240, annotation_edf[-2]["onset"] + 30 * 240)

  # Attach the annotation file to the raw edf loaded
  raw_edf.set_annotations(annotation_edf, emit_warning=False)

  # Chunk the data into 30 seconds epochs
  events, _ = mne.events_from_annotations(
      raw_edf, event_id=ANNOTATION_EVENT_ID, chunk_duration=30.0
  )


  # Create the epochs so that we can use it for classification
  tmax = 30.0 - 1.0 / raw_edf.info["sfreq"]  # tmax in included

  epochs = mne.Epochs(
      raw=raw_edf,
      events=events,
      event_id=event_id,
      tmin=0.0,
      tmax=tmax,
      baseline=None,
      preload=True,
  )

  return raw_edf, events, epochs

# The training participant will be the one with ID 0 for now
raw_train, events_train, epochs_train = load_data(participant_id=0)
# The test participant will be the one with ID 1 for now
raw_test, events_test, epochs_test = load_data(participant_id=1)

print("Training data EDF loaded and structured.")
raw_train

print("Training data epochs loaded and structured.")
epochs_train

# Plot the data
raw_train.plot(
    start=60,
    duration=120,
    scalings=dict(eeg=1e-4, resp=1e3, eog=1e-4, emg=1e-7, misc=1e-1)
)
print("Plot showing the signals form the participant 1")

# plot events across time
fig = mne.viz.plot_events(
    events_train,
    event_id=EVENT_ID,
    sfreq=raw_train.info["sfreq"],
    first_samp=events_train[0, 0],
    show=False,
)
ax = fig.gca()

# Modify the plot a bit
ax.set_title("Participant 0 Annotated Events")

# keep the color-code for further plotting
stage_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
plt.show()

# visualize participant 0 vs participant 1 PSD by sleep stage
fig, (ax1, ax2) = plt.subplots(ncols=2)

# iterate over the subjects
stages = sorted(EVENT_ID.keys())

for ax, title, epochs in zip([ax1, ax2], ["participant_0", "participant_1"], [epochs_train, epochs_test]):
  for stage, color in zip(stages, stage_colors):
    spectrum = epochs[stage].compute_psd(fmin=0.1, fmax=30.0)
    spectrum.plot(
        ci=None,
        color=color,
        axes=ax,
        show=False,
        average=True,
        amplitude=False,
        spatial_colors=False,
        picks="data",
        exclude="bads",
    )
  ax.set(title=title, xlabel="Frequency (Hz)")
ax1.set(ylabel="µV²/Hz (dB)")
ax2.legend(ax2.lines[2::3], stages)

"""# Step 2: Feature Calculation"""

def eeg_power_band(epochs, freq_bands=FREQ_BANDS):
  """Calculate relative spectral analysis on the EEG sensors for each epochs"""

  # Calculate the spectrogram
  spectrum = epochs.compute_psd(picks="eeg", fmin=0.5, fmax=30.0)
  psds, freqs = spectrum.get_data(return_freqs=True)

  # Normalization
  psds /= np.sum(psds, axis=-1, keepdims=True)

  # shape of PSDS:
  # (epoch, number of channels (we have two), frequency_bins)
  # We'll slice and average to get the delta to theta bands (5 feature per channel)
  # Therefore we should finish with (epoch, number of channels * number of bands)

  X = []
  # For each frequency band get the mean value and add it to the list X
  for fmin, fmax in freq_bands.values():
    psds_band = psds[:, :, (freqs >= fmin) & (freqs < fmax)].mean(axis=-1)
    X.append(psds_band.reshape(len(psds), -1))

  # return a numpy array, by reshuffling the list from a (5,841,2) to a (841,10)
  return np.concatenate(X, axis=1)

"""# Step 3: Classification"""

# Define beforehand what is random here for each of the stage given the data we have.
# This is for the training participant
stages = sorted(EVENT_ID.keys())
random_guess_train = {}
epochs_train.drop_bad()
for stage in stages:
  random_guess_train[stage] = len(epochs_train[stage]) / len(epochs_train)

print("Random guess for the training participant: ")
random_guess_train

random_guess_test = {}
epochs_test.drop_bad()
for stage in stages:
  random_guess_test[stage] = len(epochs_test[stage]) / len(epochs_test)

print("Random guess for the testing participant: ")
random_guess_test

# Our Steps are:
# 1. create the feature vector X using the epochs
# 2. Use the random forest classifier
pipe = make_pipeline(
    FunctionTransformer(eeg_power_band, validate=False),
    RandomForestClassifier(n_estimators=100, random_state=42)
)

# Training of the classifier using the participant 0 data
y_train = epochs_train.events[:, 2]
pipe.fit(epochs_train, y_train)

# Testing using the participant 1 data
y_pred = pipe.predict(epochs_test)

y_test = epochs_test.events[:, 2]
acc = accuracy_score(y_test, y_pred)

print(f"Accuracy score: {acc}")

"""# Step 4: Result Analysis"""

# Create a confusion matrix and a report on all the metrics
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=pipe.classes_)
disp.plot()
print(classification_report(y_test, y_pred, target_names=EVENT_ID.keys()))
print(EVENT_ID)

"""# Analysis Steps for Two Participants with One Recording for Multi State"""

# Experiment variables

# classification label target
EVENT_ID = {
    "Sleep stage W": 1,
    "Sleep stage 1": 2,
    "Sleep stage 2": 3,
    "Sleep stage 3/4": 4,
    "Sleep stage R": 5,
}


"""# Exploratory Data Anlysis of Participant 0
We will use the participant 0 for now to do the exploratory data analysis on the EEG data.

"""

def load_data(participant_id, event_id=EVENT_ID):
  """ Will load the EDF with annotation for a given participant and create
      30 seconds epochs

  Parameters
  ----------
  participant_id:
    The subjects to use. Can be in the range of 0-82 (inclusive), however the
    following subjects are not available: 39, 68, 69, 78 and 79.


  Return
  ----------
  raw_edf:
    Contains the edf with the annotations
  events:
    Contains the 30 seconds events
  epochs:
    the 30 seconds epoch

  Limitation:
  -----------
    Will only get 1 recording session
    Will only work for 1 subject at a time
  """

  ANNOTATION_EVENT_ID = {
    "Sleep stage W": 1,
    "Sleep stage 1": 2,
    "Sleep stage 2": 3,
    "Sleep stage 3": 4,
    "Sleep stage 4": 4,
    "Sleep stage R": 5,
  }

  # Load two file paths, one for the signal and one for the annotations
  [participant_file] = fetch_data(subjects=[participant_id], recording=[1])

  # Read the signal file with information on each signal
  raw_edf = mne.io.read_raw_edf(
    participant_file[0],
    stim_channel="Event marker",
    infer_types=True,
    preload=True,
    verbose="error"
  )

  # Read the annotation file
  annotation_edf = mne.read_annotations(participant_file[1])

  # keep last 30-min wake events before sleep and first 30-min wake events after
  # sleep and redefine annotations on raw data
  annotation_edf.crop(annotation_edf[1]["onset"] - 30 * 60, annotation_edf[-2]["onset"] + 30 * 60)

  # Attach the annotation file to the raw edf loaded
  raw_edf.set_annotations(annotation_edf, emit_warning=False)

  # Chunk the data into 30 seconds epochs
  events, _ = mne.events_from_annotations(
      raw_edf, event_id=ANNOTATION_EVENT_ID, chunk_duration=30.0
  )


  # Create the epochs so that we can use it for classification
  tmax = 30.0 - 1.0 / raw_edf.info["sfreq"]  # tmax in included

  epochs = mne.Epochs(
      raw=raw_edf,
      events=events,
      event_id=event_id,
      tmin=0.0,
      tmax=tmax,
      baseline=None,
  )

  return raw_edf, events, epochs

# The training participant will be the one with ID 0 for now
raw_train, events_train, epochs_train = load_data(participant_id=0)
# The test participant will be the one with ID 1 for now
raw_test, events_test, epochs_test = load_data(participant_id=1)

print("Training data EDF loaded and structured.")
raw_train

print("Training data events loaded and structured.")
events_train

print("Training data epochs loaded and structured.")
epochs_train

# Plot the data
raw_train.plot(
    start=60,
    duration=120,
    scalings=dict(eeg=1e-4, resp=1e3, eog=1e-4, emg=1e-7, misc=1e-1)
)
print("Plot showing the signals form the participant 1")

# plot events across time
fig = mne.viz.plot_events(
    events_train,
    event_id=EVENT_ID,
    sfreq=raw_train.info["sfreq"],
    first_samp=events_train[0, 0],
    show=False,
)
ax = fig.gca()

# Modify the plot a bit
ax.set_title("Participant 0 Annotated Events")

# keep the color-code for further plotting
stage_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
plt.show()

# visualize participant 0 vs participant 1 PSD by sleep stage
fig, (ax1, ax2) = plt.subplots(ncols=2)

# iterate over the subjects
stages = sorted(EVENT_ID.keys())

for ax, title, epochs in zip([ax1, ax2], ["participant_0", "participant_1"], [epochs_train, epochs_test]):
  for stage, color in zip(stages, stage_colors):
    spectrum = epochs[stage].compute_psd(fmin=0.1, fmax=30.0)
    spectrum.plot(
        ci=None,
        color=color,
        axes=ax,
        show=False,
        average=True,
        amplitude=False,
        spatial_colors=False,
        picks="data",
        exclude="bads",
    )
  ax.set(title=title, xlabel="Frequency (Hz)")
ax1.set(ylabel="µV²/Hz (dB)")
ax2.legend(ax2.lines[2::3], stages)

"""# Feature Engineering
There are multiple data type, for now we will only use EEG and only calculate power related features.
"""

def eeg_power_band(epochs, freq_bands=FREQ_BANDS):
  """Calculate relative spectral analysis on the EEG sensors for each epochs"""

  # Calculate the spectrogram
  spectrum = epochs.compute_psd(picks="eeg", fmin=0.5, fmax=30.0)
  psds, freqs = spectrum.get_data(return_freqs=True)

  # Normalization
  psds /= np.sum(psds, axis=-1, keepdims=True)

  # shape of PSDS:
  # (epoch, number of channels (we have two), frequency_bins)
  # We'll slice and average to get the delta to theta bands (5 feature per channel)
  # Therefore we should finish with (epoch, number of channels * number of bands)

  X = []
  # For each frequency band get the mean value and add it to the list X
  for fmin, fmax in freq_bands.values():
    psds_band = psds[:, :, (freqs >= fmin) & (freqs < fmax)].mean(axis=-1)
    X.append(psds_band.reshape(len(psds), -1))

  # return a numpy array, by reshuffling the list from a (5,841,2) to a (841,10)
  return np.concatenate(X, axis=1)

"""# Classification
The classification here is a very simple training with one participant and testing on another.

No cross validation going on here or parameter tuning.
"""
# Define beforehand what is random here for each of the stage given the data we have.
# This is for the training participant
stages = sorted(EVENT_ID.keys())
random_guess_train = {}
epochs_train.drop_bad()
for stage in stages:
  random_guess_train[stage] = epochs_train[stage].__len__() / epochs_train.__len__()

print("Random guess for the training participant: ")
random_guess_train

random_guess_test = {}
epochs_test.drop_bad()
for stage in stages:
  random_guess_test[stage] = epochs_test[stage].__len__() / epochs_test.__len__()

print("Random guess for the testing participant: ")
random_guess_test

# Our Steps are:
# 1. create the feature vector X using the epochs
# 2. Use the random forest classifier
pipe = make_pipeline(
    FunctionTransformer(eeg_power_band, validate=False),
    RandomForestClassifier(n_estimators=100, random_state=42)
)

# Training of the classifier using the participant 0 data
y_train = epochs_train.events[:, 2]
pipe.fit(epochs_train, y_train)

# Testing using the participant 1 data
y_pred = pipe.predict(epochs_test)

y_test = epochs_test.events[:, 2]
acc = accuracy_score(y_test, y_pred)

print(f"Accuracy score: {acc}")

# Create a confusion matrix and a report on all the metrics
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=pipe.classes_)
disp.plot()
print(classification_report(y_test, y_pred, target_names=EVENT_ID.keys()))
print(EVENT_ID)

# Save the trained model
import pickle
with open('model.pkl', 'wb') as file:
    pickle.dump(pipe, file)
print("Model saved as model.pkl")

def eeg_power_band(epochs, freq_bands=FREQ_BANDS):
    """Calculate relative spectral analysis on the EEG sensors for each epochs"""
    # Calculate the spectrogram
    spectrum = epochs.compute_psd(picks="eeg", fmin=0.5, fmax=30.0)
    psds, freqs = spectrum.get_data(return_freqs=True)
    # Normalization
    psds /= np.sum(psds, axis=-1, keepdims=True)
    # shape of PSDS:
    # (epoch, number of channels (we have two), frequency_bins)
    # We'll slice and average to get the delta to theta bands (5 feature per channel)
    # Therefore we should finish with (epoch, number of channels * number of bands)
    X = []
    # For each frequency band get the mean value and add it to the list X
    for fmin, fmax in freq_bands.values():
        psds_band = psds[:, :, (freqs >= fmin) & (freqs < fmax)].mean(axis=-1)
        X.append(psds_band.reshape(len(psds), -1))
    # return a numpy array, by reshuffling the list from a (5,841,2) to a (841,10)
    return np.concatenate(X, axis=1)