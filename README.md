# EMG Gesture Recognition Pipeline

Research software pipeline for processing electromyography (EMG) signals recorded with the COMETA acquisition system and extracting features for hand gesture recognition {open, close, rest, pinch}.

---

## Project overview

This repository contains a reproducible software pipeline for:

### OFFLINE
* Reading EMG recordings exported in ASCII format from the COMETA system.

### COMMON
* Visualizing raw EMG signals.
* Preprocessing and segmentation of signals.
* Extracting EMG features for gesture recognition.
* Supporting future gesture classification experiments.

The project has been developed as part of a Final Degree Project (TFG) focused on EMG-based hand gesture recognition.

---

## Input data

The software expects EMG recordings exported from the COMETA acquisition system in ASCII format.

Typical file structure:

* `Time(s)`
* `Emg_X(uV)`

Where `X` corresponds to the electrode/channel identifier.

Example:

```txt
Time(s)    Emg_9(uV)    Emg_10(uV)    Emg_11(uV)
0.0000     1.23         -0.55         0.11
0.0005     1.20         -0.61         0.10
```

### Recording metadata

* Device: COMETA EMG acquisition systemS
* Sampling frequency: **2000 Hz**
* Signal unit: **microvolts (uV)**
* Recording type: multi-channel EMG
* Gestures: hand opening, closing, pinch and rest 

---

## Repository structure

```txt
.
├── data/
│   ├── raw/          # Original recordings (not uploaded)
│   ├── processed/    # Generated processed signals
│   └── example/      # Example files for reproducibility
│
├── src/              # Source code
├── results/          # Generated outputs
│
├── environment.yml   # Conda environment
├── requirements.txt  # Python dependencies
├── README.md
└── LICENSE
```

---

## Installation

### 1. Clone repository

```bash
git clone <repository_url>
cd emg-gesture-pipeline
```

### 2. Create Conda environment

```bash
conda env create -f environment.yml
conda activate emg-pipeline
```

---

## Usage

Run the main pipeline:

```bash
python src/main.py
```

If you are in offline mode the software will request the input file name and generate signal visualizations and processed outputs.

---

## Current status

Current implemented functionalities:

* ASCII parser for COMETA files
* Signal loading and visualization
* EMG preprocessing
* Feature extraction

Planned work:

* Gesture classifier
* Evaluation pipeline
* ONLINE
* Additional reproducibility tooling (testing, CI, Docker)

---

## License

This software is released under the Apache 2.0 License.
