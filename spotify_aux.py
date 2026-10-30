import glob
import os
import subprocess

import librosa

from data_preparation import extract_features, preprocess_audio


def download_song(url):
    music_path = "./music_dataset/Sunnify"
    cli_path = r"C:\Users\Santiago\Downloads\sunnify-cli.exe"

    # Download Song
    command = [cli_path, "download", url]
    print("Descargando canción...")
    result = subprocess.run(command, check=False, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Error al descargar la canción: {result.stderr}")
        return None

    mp3_files = glob.glob(os.path.join(music_path, "*.mp3"))

    if not mp3_files:
        return None, None

    latest_mp3 = max(mp3_files, key=os.path.getmtime)

    y, sr = librosa.load(latest_mp3, offset=60.0, duration=30.0)

    os.remove(latest_mp3)

    return y, sr


def insert_spotify_song():
    print(
        "Ingresa el URL de la canción para su extracción. Se utilizará del minuto 1:00 a 1:30 para determinar el género"
    )
    url = input("URL: ")
    song, sr = download_song(url)

    mfcc, _ = extract_features(preprocess_audio(song), sr)

    return mfcc
