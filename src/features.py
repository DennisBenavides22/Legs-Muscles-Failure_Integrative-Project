import os
import glob
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, iirnotch

# --- 1. FILTRADO DE SEÑAL ---
def clean_emg(signal, fs=1000):
    nyq = 0.5 * fs
    b_b, a_b = butter(4, [20/nyq, 450/nyq], btype='band')
    b_n, a_n = iirnotch(60/nyq, Q=30)
    return filtfilt(b_n, a_n, filtfilt(b_b, a_b, signal))

# --- 2. EXTRACCIÓN DE CARACTERÍSTICAS PERMITIDAS ---
def extract_window_features(win_r, win_l):
    # Dominio del Tiempo (Canal Derecho)
    rms_r = np.sqrt(np.mean(win_r**2))
    mav_r = np.mean(np.abs(win_r))
    wl_r = np.sum(np.abs(np.diff(win_r)))
    var_r = np.var(win_r)
    
    # Dominio Bilateral
    rms_l = np.sqrt(np.mean(win_l**2))
    asym_ratio = rms_r / (rms_l + 1e-8)
    
    return {
        'rms_right': rms_r,
        'mav_right': mav_r,
        'wl_right': wl_r,
        'var_right': var_r,
        'asymmetry_ratio': asym_ratio
    }

# --- 3. BÚSQUEDA Y PROCESAMIENTO DE ARCHIVOS ---
data_dir = "data"
csv_files = glob.glob(os.path.join(data_dir, "*", "*.csv"))

# Identificar carpetas únicas de participantes
unique_folders = sorted(list(set([os.path.basename(os.path.dirname(f)) for f in csv_files])))
subject_map = {folder: f"SUBJ{idx+1:02d}" for idx, folder in enumerate(unique_folders)}

print(f"Se encontraron {len(unique_folders)} participantes.")

dataset_rows = []
win_len = 250   # 250 ms a 1000 Hz
step = 125      # 50% de solapamiento

for folder_name in unique_folders:
    subject_id = subject_map[folder_name]
    folder_path = os.path.join(data_dir, folder_name)
    
    # Buscar archivos correspondientes a pierna derecha e izquierda
    dr_files = sorted(glob.glob(os.path.join(folder_path, "*_dr_*.csv")))
    
    for dr_path in dr_files:
        iz_path = dr_path.replace("_dr_", "_iz_")
        
        if not os.path.exists(iz_path):
            continue
            
        try:
            df_r = pd.read_csv(dr_path)
            df_l = pd.read_csv(iz_path)
        except Exception as e:
            print(f"Error leyendo {dr_path}: {e}")
            continue

        # Obtener primera columna con datos numéricos de la señal
        sig_r_raw = df_r.select_dtypes(include=[np.number]).iloc[:, 0].values
        sig_l_raw = df_l.select_dtypes(include=[np.number]).iloc[:, 0].values
        
        # Sincronizar longitud de ambas señales
        min_len = min(len(sig_r_raw), len(sig_l_raw))
        if min_len < win_len:
            continue
            
        sig_r = clean_emg(sig_r_raw[:min_len])
        sig_l = clean_emg(sig_l_raw[:min_len])
        
        n_windows = (min_len - win_len) // step + 1
        
        for w_idx in range(n_windows):
            start = w_idx * step
            end = start + win_len
            
            win_r = sig_r[start:end]
            win_l = sig_l[start:end]
            
            feats = extract_window_features(win_r, win_l)
            feats['subject_id'] = subject_id
            
            # Etiquetado en 4 fases (0: Base, 1: Leve, 2: Mod, 3: Severa)
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

# --- 4. GUARDAR DATASET FINAL ---
if dataset_rows:
    final_df = pd.DataFrame(dataset_rows)
    os.makedirs("data", exist_ok=True)
    output_path = os.path.join("data", "features_dataset.csv")
    final_df.to_csv(output_path, index=False)
    
    print(f"\n ¡Éxito! Dataset generado con {len(final_df)} ventanas procesadas.")
    print(f" Guardado en: {output_path}")
else:
    print("\n No se procesaron ventanas. Verifica que los archivos mantengan la nomenclatura _dr_ y _iz_.")