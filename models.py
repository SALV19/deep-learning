from tensorflow.keras import Input, Sequential
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    GlobalAveragePooling2D,
    MaxPool2D,
)
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import F1Score, Precision
from tensorflow.keras.optimizers import Adam


class ModelFactory:
    @classmethod
    def compile_model(cls, input_shape, num_classes, model_type="CNN"):
        match model_type:
            case "CNN":
                return cls.__compile_cnn(input_shape, num_classes)
            case "CNN_MODIFIED":
                return cls.__compile_cnn_modified(input_shape, num_classes)

    @staticmethod
    def __compile_cnn(input_shape, num_classes):
        model = Sequential(
            [
                Input(shape=input_shape),
                Conv2D(32, (3, 3), padding="same", activation="relu"),
                MaxPool2D((2, 2)),
                Conv2D(64, (3, 3), padding="same", activation="relu"),
                MaxPool2D((2, 2)),
                Conv2D(128, (3, 3), padding="same", activation="relu"),
                GlobalAveragePooling2D(),
                Dense(64, activation="relu"),
                Dropout(0.2),
                Dense(num_classes, activation="softmax"),
            ]
        )
        model.compile(
            optimizer=Adam(),
            loss=CategoricalCrossentropy(),
            metrics=["accuracy", F1Score(average="macro"), Precision()],
        )
        return model

    @staticmethod
    def __compile_cnn_modified(input_shape, num_classes):
        raise ValueError("Method not yet implementes")
