import os
import pandas as pd
import numpy as np
from datasets import load_dataset, Audio # Huggingface - lewtun/music_genres
import librosa
import matplotlib.pyplot as plt
from data_preparation import preprocess_audio, extract_features

# Loading the data
path = "./music_dataset/gztan/"

script_dir = os.path.dirname(__file__)
graph_dir = os.path.join(script_dir, 'graphs/')
data_dir = os.path.join(script_dir, 'data/')


records = []
corrupted = []

for root, _, files in os.walk(path):
    for file in files:
        try: 
          if file.endswith(".wav"):
            file_path = os.path.join(root, file)
            audio, _ = librosa.load(file_path, sr=22050)
            genre = os.path.basename(root)
            
            records.append({
              "audio": audio,
              "genre": genre,
              "file": file_path
            })
        except:
          corrupted.append(file_path)
          print(f"Skipping corrupted file: {file_path}")

# Construct DataFrame all at once
gztan = pd.DataFrame(records)

# Displaying sound wave
sr = 22050 

audio_samples = gztan.sample(frac=1).head(4)

fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(12, 6))

for idx, (_, audio) in enumerate(audio_samples.iterrows()):
    signal = audio["audio"]
    times = np.arange(len(signal)) / sr
    
    curr_ax = ax[idx // 2, idx % 2]
    curr_ax.plot(times, signal)
    
    curr_ax.set(title=f"Audio Genre: {audio['genre']}", xlabel="Time (s)", ylabel="Amplitude")

plt.subplots_adjust(hspace=0.4, wspace=0.3)
plt.tight_layout()
plt.savefig(graph_dir + "audio_samples")

gztan.to_pickle(data_dir + "gztan.pkl")

# Display FFT and MFCC
audio_samples = gztan.sample(n=4, random_state=42)

fig_mfcc, ax_mfcc = plt.subplots(2, 2, figsize=(12, 8))
fig_fft, ax_fft = plt.subplots(2, 2, figsize=(12, 8))

for idx, (_, audio) in enumerate(audio_samples.iterrows()):
    # Preprocesar audio y extraer características
    signal = preprocess_audio(audio["audio"])
    mfcc, fft = extract_features(signal)

    # Posición en la cuadrícula
    row, col = divmod(idx, 2)

    # Gráfica MFCC
    curr_ax = ax_mfcc[row, col]
    curr_ax.imshow(
        mfcc,
        aspect="auto",
        origin="lower",
        interpolation="nearest"
    )
    curr_ax.set(
        title=f"MFCC - Genre: {audio['genre']}",
        xlabel="Time Frames",
        ylabel="MFCC Coefficients"
    )

    # Gráfica FFT
    curr_ax = ax_fft[row, col]

    freqs = np.linspace(0, sr / 2, len(fft))
    curr_ax.plot(freqs, fft)
    curr_ax.set_xlabel("Frequency (Hz)")
    curr_ax.set_ylabel("Magnitude")

    curr_ax.set_title(f"FFT - Genre: {audio['genre']}")

# Ajustar y guardar MFCC
fig_mfcc.suptitle("MFCC Features — 4 Audio Samples", fontsize=14)
fig_mfcc.tight_layout()
fig_mfcc.savefig(graph_dir + "mfcc_samples.png", dpi=150, bbox_inches="tight")

# Ajustar y guardar FFT
fig_fft.suptitle("FFT Features — 4 Audio Samples", fontsize=14)
fig_fft.tight_layout()
fig_fft.savefig(graph_dir + "fft_samples.png", dpi=150, bbox_inches="tight")

plt.show()