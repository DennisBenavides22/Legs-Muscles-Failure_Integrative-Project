import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- 1. LOCALIZAR ARCHIVO DE DATOS ---
project_root = os.getcwd()

# Intentar cargar primero el dataset de características procesado
dataset_path = os.path.join("data", "features_dataset.csv")

if os.path.exists(dataset_path):
    print(f"Cargando dataset procesado desde: {dataset_path}")
    df = pd.read_csv(dataset_path)
    
    # Graficar la evolución temporal del RMS y la Varianza por participante
    subject = df['subject_id'].unique()[0]
    subj_df = df[df['subject_id'] == subject].reset_index(drop=True)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    
    # Gráfica 1: RMS Pierna Derecha
    ax1.plot(subj_df.index, subj_df['rms_right'], color='crimson', label='RMS Pierna Derecha')
    ax1.set_title(f"Evolución de Características EMG durante la Fatiga — {subject}")
    ax1.set_ylabel("RMS")
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(loc='upper left')
    
    # Gráfica 2: Ratio de Asimetría
    ax2.plot(subj_df.index, subj_df['asymmetry_ratio'], color='navy', label='Ratio de Asimetría (Der/Izq)')
    ax2.set_xlabel("Índice de Ventana (Tiempo)")
    ax2.set_ylabel("Asymmetry Ratio")
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend(loc='upper left')
    
    plt.tight_layout()
    
    os.makedirs("results", exist_ok=True)
    save_path = os.path.join("results", "features_progression_plot.png")
    plt.savefig(save_path, dpi=300)
    plt.show()
    print(f" Gráfica guardada en: {save_path}")

else:
    # Si no existe features_dataset.csv, buscar cualquier CSV en la carpeta data
    csv_files = glob.glob(os.path.join(project_root, "**", "*.csv"), recursive=True)
    raw_files = [f for f in csv_files if "results" not in f]
    
    if not raw_files:
        print(" No se encontró 'features_dataset.csv' ni ningún archivo .csv en la carpeta data/.")
        print("Asegúrate de ejecutar desde la raíz del proyecto o colocar 'features_dataset.csv' en la carpeta data/.")
    else:
        sample_file = raw_files[0]
        print(f"Graficando señal desde: {sample_file}")
        raw_df = pd.read_csv(sample_file)
        signal = raw_df.select_dtypes(include=[np.number]).iloc[:, 0].values
        
        plt.figure(figsize=(11, 4))
        plt.plot(signal, color='navy', linewidth=0.8)
        plt.title(f"Señal EMG Bruta — {os.path.basename(sample_file)}")
        plt.xlabel("Muestras")
        plt.ylabel("Amplitud")
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()