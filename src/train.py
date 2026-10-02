"""Entrena VGG-16 (transfer learning) para distinguir <STUDENT_NAME> vs Fondo.

Uso:
    python src/train.py --epochs 30 --batch-size 16 --optimizer adam --lr 1e-4
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input

from config import CLASS_NAMES, CLASSES_PATH, DATA_DIR, IMG_SIZE, MODEL_DIR, MODEL_PATH, REPORT_DIR


def load_datasets(batch_size, val_split, seed):
    kw = dict(
        directory=DATA_DIR,
        class_names=CLASS_NAMES,
        label_mode="categorical",
        image_size=IMG_SIZE,
        batch_size=batch_size,
        validation_split=val_split,
        seed=seed,
    )
    train = keras.utils.image_dataset_from_directory(subset="training", **kw)
    val = keras.utils.image_dataset_from_directory(subset="validation", shuffle=False, **kw)
    return train, val


def build_model(lr, optimizer_name, loss):
    # Aumento de datos: solo se aplica en entrenamiento.
    augment = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.08),
        layers.RandomZoom(0.15),
        layers.RandomBrightness(0.2),
        layers.RandomContrast(0.2),
    ], name="augment")

    base = VGG16(weights="imagenet", include_top=False, input_shape=IMG_SIZE + (3,))
    base.trainable = False

    inputs = keras.Input(shape=IMG_SIZE + (3,))
    x = augment(inputs)
    x = layers.Lambda(preprocess_input, name="vgg_preprocess")(x)  # RGB -> formato VGG16
    x = base(x, training=False)
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(len(CLASS_NAMES), activation="softmax")(x)
    model = keras.Model(inputs, outputs)

    opt = {"adam": keras.optimizers.Adam, "sgd": keras.optimizers.SGD,
           "rmsprop": keras.optimizers.RMSprop}[optimizer_name](learning_rate=lr)
    model.compile(loss=loss, optimizer=opt, metrics=["accuracy"])
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--optimizer", default="adam", choices=["adam", "sgd", "rmsprop"])
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--loss", default="categorical_crossentropy")
    ap.add_argument("--val-split", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    MODEL_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)

    train, val = load_datasets(args.batch_size, args.val_split, args.seed)
    train, val = train.prefetch(tf.data.AUTOTUNE), val.prefetch(tf.data.AUTOTUNE)

    # Pesos por clase, por si hay mas fondos que fotos tuyas (o al reves).
    counts = np.array([len(list((DATA_DIR / c).glob("*"))) for c in CLASS_NAMES], dtype=float)
    class_weight = {i: counts.sum() / (len(counts) * n) for i, n in enumerate(counts)}
    print("Imagenes por clase:", dict(zip(CLASS_NAMES, counts.astype(int))), "| pesos:", class_weight)

    model = build_model(args.lr, args.optimizer, args.loss)
    cbs = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(MODEL_PATH, monitor="val_loss", save_best_only=True),
    ]
    hist = model.fit(train, validation_data=val, epochs=args.epochs,
                     class_weight=class_weight, callbacks=cbs)

    # Evaluacion final con el mejor modelo
    model = keras.models.load_model(MODEL_PATH)
    y_true = np.concatenate([np.argmax(y, axis=1) for _, y in val])
    y_pred = np.argmax(model.predict(val), axis=1)
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))
    print("Matriz de confusion:\n", confusion_matrix(y_true, y_pred))

    CLASSES_PATH.write_text(json.dumps(CLASS_NAMES, ensure_ascii=False))

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for a, k, t in zip(ax, ["loss", "accuracy"], ["Perdida", "Exactitud"]):
        a.plot(hist.history[k], label="entrenamiento")
        a.plot(hist.history["val_" + k], label="validacion")
        a.set_title(t); a.set_xlabel("epoca"); a.legend()
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "curvas_entrenamiento.png", dpi=120)
    print("Modelo guardado en", MODEL_PATH)


if __name__ == "__main__":
    main()
