import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, ConfusionMatrixDisplay



loss, test_accuracy, f1_score, precision = model.evaluate(X_test, y_test, verbose=0)

print(f"Test loss: {loss:.4f}")
print(f"Test accuracy: {test_accuracy:.4f}")
print("Test f1:", f1_score)
print(f"Test precision: {precision:.4f}")

y_pred = np.argmax(model.predict(X_test), axis=1)
y_test_classes = np.argmax(y_test, axis=1)

print(
    classification_report(
        y_test_classes,  # Usa la versión 1D de y_test
        y_pred,
        labels=np.arange(len(label_encoder.classes_)),
        target_names=label_encoder.classes_,
        zero_division=0,
    )
)

# Matriz de confusión
ConfusionMatrixDisplay.from_predictions(
    y_test_classes,  # Usa la versión 1D de y_test
    y_pred,
    labels=np.arange(len(label_encoder.classes_)),
    display_labels=label_encoder.classes_,
    xticks_rotation=45,
    cmap="Blues",
)

plt.tight_layout()
plt.show()