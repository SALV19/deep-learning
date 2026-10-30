from pathlib import Path

import numpy as np
import tensorflow as tf

from spotify_aux import insert_spotify_song

# Load trained model
MODEL_PATH = "model/best_genre_model.keras"

model = tf.keras.models.load_model(MODEL_PATH)

GENRES = [
    "blues",
    "classical",
    "country",
    "disco",
    "hiphop",
    "jazz",
    "metal",
    "pop",
    "reggae",
    "rock",
]


def predict_genre(input_data):
    """
    input_data: Preprocessed audio data.
    Must have the same shape and format used during training.
    """

    # Run prediction
    predictions = model.predict(input_data, verbose=0)[0]

    # Get predicted genre
    predicted_index = np.argmax(predictions)
    predicted_genre = GENRES[predicted_index]

    print(f"\nPredicted genre: {predicted_genre}")
    print("\nProbabilities:")

    for genre, probability in zip(GENRES, predictions):
        print(f"{genre:12s}: {probability * 100:.2f}%")

    return predicted_genre, predictions


def get_input():
    print(
        "Selecciona una canción predefinida o ingresa un url de spotify para determinar el género de la canción"
    )
    print(
        "\nNota: Las canciones propuestas vienen del set de pruebas que se utilizó para evaluar el modelo, por lo cual no pasaron por el set de entrenamiento"
    )
    while True:
        for idx, file in enumerate(files):
            print(f"{idx + 1}. {file.__str__().split('\\')[-1].replace('.npz', '')}")
        print(f"{len(files) + 1}. Subir canción de spotify")

        selection = int(input("\nSelección: "))

        if selection < len(files) + 1 and selection > 0:
            print(files[selection - 1])
            return np.load(files[selection - 1])["mfcc"]
        elif selection == len(files) + 1:
            return insert_spotify_song()


if __name__ == "__main__":
    # TODO: Replace this with your preprocessing output
    # Example:
    # processed_audio = preprocess_audio("music.wav")
    script_dir = Path(__file__).resolve().parent
    test_dir = script_dir / "test"
    files = sorted(test_dir.glob("*.npz"))

    music_mfcc = get_input()
    # Fit dimensions: (X, Y) -> (None, X, Y, 1)
    input_data = np.expand_dims(music_mfcc, axis=-1)
    input_data = np.expand_dims(input_data, axis=0)

    predict_genre(input_data)
