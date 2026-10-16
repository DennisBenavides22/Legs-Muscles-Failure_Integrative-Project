import os
import glob
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, iirnotch, hilbert, find_peaks
from scipy.interpolate import interp1d
from scipy.stats import pearsonr

# --- 1. SIGNAL FILTERING & ENVELOPE EXTRACTION ---
def clean_emg(signal, fs=1000):
    """Applies 4th-order 20-450 Hz Butterworth bandpass & 60 Hz notch filters."""
    nyq = 0.5 * fs
    b_b, a_b = butter(4, [20/nyq, 450/nyq], btype='band')
    b_n, a_n = iirnotch(60/nyq, Q=30)
    return filtfilt(b_n, a_n, filtfilt(b_b, a_b, signal))

def get_linear_envelope(signal, fs=1000, cutoff=3):
    """Computes the Hilbert linear envelope smoothed with a 3 Hz low-pass filter."""
    analytic_signal = hilbert(signal)
    envelope = np.abs(analytic_signal)
    nyq = 0.5 * fs
    b, a = butter(2, cutoff / nyq, btype='low')
    return filtfilt(b, a, envelope)

# --- 2. SECTION 2.3 DATA CURATION & REPETITION FILTERING ---
def curate_and_sync_signals(sig_r, sig_l, fs=1000, min_corr=0.65):
    """
    Automated Section 2.3 Exclusion Pipeline:
    1. Trims pre-exercise and post-exercise resting baselines.
    2. Segments repetition contraction bursts.
    3. Builds an intra-subject ensemble average template normalized to 0-100% cycle.
    4. Excludes anomalous repetitions (Pearson r < 0.65) while preserving bilateral symmetry.
    """
    env_r = get_linear_envelope(sig_r, fs=fs)
    env_l = get_linear_envelope(sig_l, fs=fs)
    combined_env = (env_r + env_l) / 2.0
    
    # Adaptive threshold for burst detection
    threshold = np.mean(combined_env) + 0.4 * np.std(combined_env)
    min_distance = int(0.8 * fs)  # Minimum 0.8s between repetitions
    
    peaks, _ = find_peaks(combined_env, height=threshold, distance=min_distance)
    
    # Fallback if insufficient repetitions detected
    if len(peaks) < 2:
        return sig_r, sig_l

    rep_bounds = []
    rep_segments_r = []
    
    for p in peaks:
        left = p
        while left > 0 and combined_env[left] > 0.3 * threshold:
            left -= 1
        right = p
        while right < len(combined_env) - 1 and combined_env[right] > 0.3 * threshold:
            right += 1
            
        dur = (right - left) / fs
        if 0.5 <= dur <= 6.0:  # Valid repetition duration (0.5s to 6.0s)
            rep_bounds.append((left, right))
            rep_segments_r.append(env_r[left:right])

    if not rep_bounds:
        return sig_r, sig_l

    # 1. Baseline Trimming: Extract active exercise interval (first onset to last offset)
    first_onset = rep_bounds[0][0]
    last_offset = rep_bounds[-1][1]
    
    # 2. Time-normalize reps to 100 points for ensemble template creation
    norm_reps_r = []
    for seg in rep_segments_r:
        x_orig = np.linspace(0, 1, len(seg))
        f_interp = interp1d(x_orig, seg, kind='linear')
        norm_reps_r.append(f_interp(np.linspace(0, 1, 100)))

    # 3. Compute Intra-Subject Ensemble Template
    template_r = np.mean(norm_reps_r, axis=0)

    # 4. Correlation Filtering (Pearson r >= 0.65)
    valid_indices = []
    for idx, norm_rep in enumerate(norm_reps_r):
        r_val, _ = pearsonr(norm_rep, template_r)
        if np.isnan(r_val) or r_val >= min_corr:
            valid_indices.append(idx)

    # If all repetitions fail or pass, return trimmed active region
    if not valid_indices or len(valid_indices) == len(rep_bounds):
        return sig_r[first_onset:last_offset], sig_l[first_onset:last_offset]

    # Mask out invalid rep intervals within the active region
    keep_mask = np.ones(last_offset - first_onset, dtype=bool)
    for idx, (l, r) in enumerate(rep_bounds):
        if idx not in valid_indices:
            keep_mask[l - first_onset : r - first_onset] = False

    curated_r = sig_r[first_onset:last_offset][keep_mask]
    curated_l = sig_l[first_onset:last_offset][keep_mask]

    return curated_r, curated_l

# --- 3. TIME-DOMAIN FEATURE EXTRACTION ---
def extract_window_features(win_r, win_l):
    # Time Domain Features (Right Leg)
    rms_r = np.sqrt(np.mean(win_r**2))
    mav_r = np.mean(np.abs(win_r))
    wl_r = np.sum(np.abs(np.diff(win_r)))
    var_r = np.var(win_r)
    
    # Bilateral Feature (Left Leg comparison)
    rms_l = np.sqrt(np.mean(win_l**2))
    
    # Robust Asymmetry Ratio with epsilon and clipping (prevents extreme outliers)
    asym_ratio = (rms_r + 1e-6) / (rms_l + 1e-6)
    asym_ratio = np.clip(asym_ratio, 0.1, 10.0)
    
    return {
        'rms_right': rms_r,
        'mav_right': mav_r,
        'wl_right': wl_r,
        'var_right': var_r,
        'asymmetry_ratio': asym_ratio
    }

# --- 4. FILE PROCESSING & DATASET GENERATION ---
data_dir = "data"
csv_files = glob.glob(os.path.join(data_dir, "*", "*.csv"))

# Map participant folders to anonymized SUBJxx IDs
unique_folders = sorted(list(set([os.path.basename(os.path.dirname(f)) for f in csv_files])))
subject_map = {folder: f"SUBJ{idx+1:02d}" for idx, folder in enumerate(unique_folders)}

print(f"Se encontraron {len(unique_folders)} participantes.")
print("Aplicando pipeline de curación (Recorte de tiempos muertos y filtro de plantilla ensemble r >= 0.65)...")

dataset_rows = []
win_len = 250   # 250 ms windows at 1000 Hz
step = 125      # 50% overlap

for folder_name in unique_folders:
    subject_id = subject_map[folder_name]
    folder_path = os.path.join(data_dir, folder_name)
    
    dr_files = sorted(glob.glob(os.path.join(folder_path, "*_dr_*.csv")))
    
    for dr_path in dr_files:
        iz_path = dr_path.replace("_dr_", "_iz_")
        
        if not os.path.exists(iz_path):
            continue
            
        # Skip MVC calibration files
        if "mvc" in os.path.basename(dr_path).lower():
            continue

        try:
            df_r = pd.read_csv(dr_path)
            df_l = pd.read_csv(iz_path)
        except Exception as e:
            print(f"Error leyendo {dr_path}: {e}")
            continue

        # Extract numerical raw signals
        sig_r_raw = df_r.select_dtypes(include=[np.number]).iloc[:, 0].values
        sig_l_raw = df_l.select_dtypes(include=[np.number]).iloc[:, 0].values
        
        min_len = min(len(sig_r_raw), len(sig_l_raw))
        if min_len < win_len:
            continue
            
        # 1. Bandpass & Notch Filtering
        sig_r_clean = clean_emg(sig_r_raw[:min_len])
        sig_l_clean = clean_emg(sig_l_raw[:min_len])
        
        # 2. Section 2.3 Repetition Curation & Baseline Trimming
        sig_r, sig_l = curate_and_sync_signals(sig_r_clean, sig_l_clean, min_corr=0.65)
        
        curated_len = min(len(sig_r), len(sig_l))
        if curated_len < win_len:
            continue
            
        n_windows = (curated_len - win_len) // step + 1
        
        # 3. Sliding Window Extraction & Temporal Labeling
        for w_idx in range(n_windows):
            start = w_idx * step
            end = start + win_len
            
            win_r = sig_r[start:end]
            win_l = sig_l[start:end]
            
            feats = extract_window_features(win_r, win_l)
            feats['subject_id'] = subject_id
            
            # Temporal phase labeling (0: Baseline, 1: Early, 2: Moderate, 3: Severe)
            progress = w_idx / float(n_windows)
            if progress < 0.25:
                label = 0
            elif progress < 0.50:
                label = 1
            elif progress < 0.75:
                label = 2
            else:
                label = 3
                
            feats['label_phase'] = label
            dataset_rows.append(feats)

# --- 5. EXPORT DATASET ---
if dataset_rows:
    final_df = pd.DataFrame(dataset_rows)
    os.makedirs("data", exist_ok=True)
    output_path = os.path.join("data", "features_dataset.csv")
    final_df.to_csv(output_path, index=False)
    
    print(f"\n ¡Éxito! Dataset curado y generado con {len(final_df)} ventanas procesadas.")
    print(f" Guardado en: {output_path}")
else:
    print("\n No se procesaron ventanas. Revisa la ubicación y nomenclatura de los archivos.")