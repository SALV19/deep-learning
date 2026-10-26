import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import F1Score, Precision
from tensorflow.keras import Input
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical


# Load data

# Split train-validation-test
X = np.stack(gztan["mfcc"].to_numpy())
X = X[..., np.newaxis]

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(gztan["genre"])
y = to_categorical(y, num_classes=10)

# Split 70:30
X_train, X_temp, y_train, y_temp = train_test_split(
  X, y,
  test_size=0.30,
  random_state=42,
  stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
  X_temp, y_temp,
  test_size=.5,
  random_state=42,
  stratify=y_temp
)

print("Train:", X_train.shape, y_train.shape)
print("Val:", X_val.shape, y_val.shape)
print("Test:", X_test.shape, y_test.shape)
print("Genres:", label_encoder.classes_)


# Normalización
mean = X_train.mean(axis=(0,2), keepdims=True)
std = X_train.std(axis=(0,2), keepdims=True)
std = np.maximum(std, 1e-6) # Evitar división entre 0

X_train = (X_train - mean) / std
X_val = (X_val - mean) / std
X_test = (X_test - mean) / std

# Creación del Modelo CNN
def compile_model(input_shape, num_clases):
  model = Sequential([
    Input(shape=input_shape),
    
    Conv2D(32, (3, 3), padding="same", activation="relu"),
    MaxPool2D((2,2)),
    
    Conv2D(64, (3,3), padding="same", activation="relu"),
    MaxPool2D((2,2)),
    
    Conv2D(128, (3,3), padding="same", activation="relu"),
    GlobalAveragePooling2D(),
    
    Dense(64, activation="relu"),
    Dropout(0.2),
    
    Dense(num_clases, activation="softmax")
  ])
  
  model.compile( 
    optimizer=Adam(), 
    loss=CategoricalCrossentropy(),
    metrics=["accuracy", F1Score(), Precision()]
  )
  
  return model
  
model = compile_model(
  input_shape=X_train.shape[1:],
  num_clases=len(label_encoder.classes_)
)

model.summary()

# Entrenamiento
callbacks = [
  EarlyStopping(
    monitor="val_loss",
    patience=8,
    restore_best_weights=True
  ),
  ModelCheckpoint(
    "best_genre_model.keras",
    monitor="val_loss",
    save_best_only=True
  )
]

history = model.fit(
  X_train, y_train, 
  validation_data=(X_val, y_val),
  epochs=50,
  batch_size=16,
  callbacks=callbacks,
  verbose=1
)