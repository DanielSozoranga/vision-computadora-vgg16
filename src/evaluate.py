"""Evalua el modelo guardado con el conjunto de validacion (el mismo 20 % que usa train.py).

Imprime el reporte de clasificacion y la matriz de confusion, y guarda la matriz como imagen en
reports/matriz_confusion.png.

Uso:
    python src/evaluate.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow import keras

import vgg_layer  # noqa: F401  (registra la capa VGGPreprocess)
from config import CLASS_NAMES, MODEL_PATH, REPORT_DIR
from train import load_datasets


def main():
    model = keras.models.load_model(MODEL_PATH)
    print("Modelo cargado desde", MODEL_PATH)

    _, val = load_datasets(batch_size=32, val_split=0.2, seed=42)
    batches = list(val)
    x_val = np.concatenate([x.numpy() for x, _ in batches])
    y_true = np.concatenate([np.argmax(y.numpy(), axis=1) for _, y in batches])
    y_pred = np.argmax(model.predict(x_val, verbose=0), axis=1)

    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))
    cm = confusion_matrix(y_true, y_pred)
    print("Matriz de confusion:\n", cm)
    print(f"Exactitud: {(y_true == y_pred).mean():.1%}")

    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(CLASS_NAMES)), CLASS_NAMES)
    ax.set_yticks(range(len(CLASS_NAMES)), CLASS_NAMES)
    ax.set_xlabel("Prediccion")
    ax.set_ylabel("Real")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)
    fig.tight_layout()
    REPORT_DIR.mkdir(exist_ok=True)
    fig.savefig(REPORT_DIR / "matriz_confusion.png", dpi=130)
    print("Imagen guardada en reports/matriz_confusion.png")


if __name__ == "__main__":
    main()
