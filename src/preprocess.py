import os
import glob
import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt, iirnotch

# --- 1. DEFINE FILTERING FUNCTION ---
def clean_emg_signal(signal, fs=1000):
    nyquist = 0.5 * fs
    # 20-450 Hz Bandpass filter
    b_band, a_band = butter(4, [20 / nyquist, 450 / nyquist], btype='band')
    bandpassed = filtfilt(b_band, a_band, signal)
    
    # 60 Hz Powerline Notch filter
    b_notch, a_notch = iirnotch(60 / nyquist, Q=30)
    cleaned = filtfilt(b_notch, a_notch, bandpassed)
    
    return cleaned

# --- 2. FIND ALL CSV FILES IN SUBFOLDERS ---
data_dir = "data"
csv_files = glob.glob(os.path.join(data_dir, "*", "*.csv"))

# Get unique student folder names (e.g., Dennis_Benavides)
unique_folders = sorted(list(set([f.split(os.sep)[-2] for f in csv_files])))

# Create an anonymization mapping dictionary: {'Dennis_Benavides': 'SUBJ01', ...}
subject_map = {folder_name: f"SUBJ{idx+1:02d}" for idx, folder_name in enumerate(unique_folders)}

print(f"Found {len(unique_folders)} participant folders across {len(csv_files)} CSV files.")
print("Anonymization Mapping Created:")
for real_name, anon_id in subject_map.items():
    print(f"  {real_name} -> {anon_id}")

# --- 3. PROCESS EACH FILE IN A LOOP ---
for file_path in csv_files:
    parts = file_path.split(os.sep)
    real_folder_name = parts[-2]
    anon_subject_id = subject_map[real_folder_name]  # Uses anonymized ID
    file_name = parts[-1]
    
    print(f"Processing: Subject={anon_subject_id} | File={file_name}")
    
    # Load raw data
    df = pd.read_csv(file_path)
    
    # Filter right and left channels if they exist
    if 'ch_right' in df.columns:
        clean_right = clean_emg_signal(df['ch_right'].values)
    if 'ch_left' in df.columns:
        clean_left = clean_emg_signal(df['ch_left'].values)

print("Batch preprocessing complete!")