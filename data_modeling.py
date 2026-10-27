from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from models import ModelFactory

BATCH_SIZE = 16

# Dir
script_dir = Path(__file__).resolve().parent

audio_dir = script_dir / "data" / "audio_batches"
graph_dir = script_dir / "graphs"

output_dir = script_dir / "model"
output_dir.mkdir(parents=True, exist_ok=True)

# Load data & split train_val_test
files = sorted(audio_dir.glob("batch_*.npz"))
files_train, files_temp = train_test_split(files, test_size=0.3, random_state=42)
files_val, files_test = train_test_split(
    files_temp,
    test_size=0.5,
    random_state=42,
)

# Genre enconding and normalization
count = 0
s = None
s2 = None

all_genres = []
for f in files:
    with np.load(f, allow_pickle=True) as d:
        all_genres.extend(d["genre"])
        x = d["mfcc"].astype(np.float64)
    if s is None:
        s = np.zeros(x.shape[1], dtype=np.float64)
        s2 = np.zeros(x.shape[1], dtype=np.float64)
    s += x.sum(axis=(0, 2))
    s2 += (x**2).sum(axis=(0, 2))
    count += x.shape[0] * x.shape[2]

label_encoder = LabelEncoder().fit(all_genres)
NUM_CLASSES = len(label_encoder.classes_)

mean = s / count
std = np.sqrt(np.maximum(s2 / count - mean**2, 1e-12))
std = np.maximum(std, 1e-6)
mean = mean.reshape(1, -1, 1).astype(np.float32)
std = std.reshape(1, -1, 1).astype(np.float32)

np.savez(output_dir / "norm_stats.npz", mean=mean, std=std)


# Generator to extract one batch at a time
def make_generator(file_list):
    def gen():
        order = list(file_list)
        np.random.shuffle(order)
        for f in order:
            with np.load(f, allow_pickle=True) as d:
                X = d["mfcc"].astype(np.float32)
                y = label_encoder.transform(d["genre"])
            X = ((X - mean) / std)[..., np.newaxis]
            y = tf.keras.utils.to_categorical(y, NUM_CLASSES).astype(np.float32)
            for i in range(len(X)):
                yield X[i], y[i]

    return gen


with np.load(files_train[0], allow_pickle=True) as d:
    sample_shape = d["mfcc"].shape[1:] + (1,)

signature = (
    tf.TensorSpec(shape=sample_shape, dtype=tf.float32),
    tf.TensorSpec(shape=(NUM_CLASSES,), dtype=tf.float32),
)


def make_dataset(file_list):
    ds = tf.data.Dataset.from_generator(
        make_generator(file_list), output_signature=signature
    )
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


# Loading Datasets (Batches)
train_ds = make_dataset(files_train)
val_ds = make_dataset(files_val)
test_ds = make_dataset(files_test)

model = ModelFactory.compile_model(sample_shape, NUM_CLASSES)
model.summary()

callbacks = [
    EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
    ModelCheckpoint(
        output_dir / "best_genre_model.keras", monitor="val_loss", save_best_only=True
    ),
]

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=50,
    callbacks=callbacks,
    verbose=1,
)

loss, test_accuracy, f1_score, precision = model.evaluate(test_ds, verbose=0)

print(f"Test loss: {loss:.4f}")
print(f"Test accuracy: {test_accuracy:.4f}")
print(f"Test f1 (macro): {float(f1_score):.4f}")
print(f"Test precision: {precision:.4f}")

y_true_list = []
y_pred_list = []

for X_batch, y_batch in test_ds:
    probs = model.predict_on_batch(X_batch)
    y_pred_list.append(np.argmax(probs, axis=1))
    y_true_list.append(np.argmax(y_batch.numpy(), axis=1))

y_pred = np.concatenate(y_pred_list)
y_test_classes = np.concatenate(y_true_list)

print("Test samples:", len(y_test_classes))

print(
    classification_report(
        y_test_classes,
        y_pred,
        labels=np.arange(len(label_encoder.classes_)),
        target_names=label_encoder.classes_,
        zero_division=0,
    )
)

ConfusionMatrixDisplay.from_predictions(
    y_test_classes,
    y_pred,
    labels=np.arange(len(label_encoder.classes_)),
    display_labels=label_encoder.classes_,
    xticks_rotation=45,
    cmap="Blues",
)

plt.tight_layout()
plt.savefig(graph_dir / "model_evaluation.png")
plt.show()

plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.title("model accuracy")
plt.ylabel("accuracy")
plt.xlabel("epoch")
plt.legend(["train", "val"], loc="upper left")
plt.savefig(graph_dir / "model_accuracy.png")
plt.show()

plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("model loss")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend(["train", "val"], loc="upper left")
plt.savefig(graph_dir / "model_loss.png")
plt.show()
