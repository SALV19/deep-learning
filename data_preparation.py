import random
from pathlib import Path

import librosa
import numpy as np

script_dir = Path(__file__).resolve().parent

output_dir = script_dir / "data" / "audio_batches"
output_dir.mkdir(parents=True, exist_ok=True)

# Load data
audio_dir = script_dir / "music_dataset" / "gztan"

# Process data to MFCC
SR = 22050
N_FFT = 2048
N_MFCC = 13
HOP_LENGTH = 512
DURATION = 30  # segundos
N_SAMPLES = SR * DURATION  # 661500
BATCH_SIZE = 50

files = sorted(audio_dir.glob("*/*.wav"))
print(f"Canciones encontradas: {len(files)}")
print(f"Expected output files: {len(files) / BATCH_SIZE}")

random.Random(42).shuffle(files)

mfcc_features = []
fft_features = []
labels = []
paths = []
batch_number = 0


# Process audio into MFCC
def preprocess_audio(audio):
    audio = np.asarray(audio, dtype=np.float32)

    if len(audio) > N_SAMPLES:
        audio = audio[:N_SAMPLES]
    else:
        audio = np.pad(audio, (0, N_SAMPLES - len(audio)))

    return audio


def extract_features(audio, sr=SR):
    """Extract MFCCs and FFT magnitude from one audio file."""
    mfcc = librosa.feature.mfcc(
        y=audio, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mfcc=N_MFCC
    ).astype(np.float32)

    # STFT: complex frequency coefficients over time
    stft = librosa.stft(y=audio, n_fft=N_FFT, hop_length=HOP_LENGTH)

    # Magnitude spectrum: shape (N_FFT // 2 + 1, number_of_frames)
    fft_magnitude = np.abs(stft).astype(np.float32)

    return mfcc, fft_magnitude


# Save batch to folder
def save_batch():
    global batch_number

    if not labels:
        return

    output_path = output_dir / f"batch_{batch_number:03d}.npz"

    np.savez_compressed(
        output_path,
        mfcc=np.stack(mfcc_features),
        fft=np.stack(fft_features),
        genre=np.asarray(labels),
        paths=np.asarray(paths),
    )

    print(f"Saved {output_path.name}: {len(mfcc_features)} songs")

    mfcc_features.clear()
    fft_features.clear()
    labels.clear()
    paths.clear()
    batch_number += 1


# Use batches to load and save data
def main():
    for i, file_path in enumerate(files, start=1):
        try:
            # Load only one audio file into memory
            audio, _ = librosa.load(file_path, sr=SR, mono=True, duration=DURATION)

            audio = preprocess_audio(audio)
            mfcc, fft = extract_features(audio)

            mfcc_features.append(mfcc)
            fft_features.append(fft)
            labels.append(file_path.parent.name)
            paths.append(str(file_path.relative_to(audio_dir)))

            del audio, mfcc, fft

            if len(mfcc_features) >= BATCH_SIZE:
                save_batch()

        except Exception as e:
            print(f"Skipping corrupted file: {file_path} ({e})")

    # Save the final batch, even if it contains fewer than 25 songs
    save_batch()

    print("Processing complete.")


if __name__ == "__main__":
    main()
