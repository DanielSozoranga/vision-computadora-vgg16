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

from vgg_layer import VGGPreprocess

from config import CLASS_NAMES, CLASSES_PATH, IMG_SIZE, MODEL_DIR, MODEL_PATH, PROCESSED_DIR, REPORT_DIR

DATA_DIR = PROCESSED_DIR   # se entrena con los rostros recortados (python src/crop_faces.py)


def load_datasets(batch_size, val_split, seed):
    kw = dict(
        directory=DATA_DIR,
        class_names=CLASS_NAMES,
        label_mode="categorical",
        image_size=IMG_SIZE,
        crop_to_aspect_ratio=True,   # recorta al centro en vez de deformar fotos verticales/horizontales
        batch_size=batch_size,
        validation_split=val_split,
        seed=seed,
    )
    train = keras.utils.image_dataset_from_directory(subset="training", **kw)
    # Con shuffle=False la validacion saldria solo de la ultima clase (archivos ordenados por carpeta).
    # Con la misma seed y shuffle=True, train y validacion quedan separados y con las 2 clases.
    val = keras.utils.image_dataset_from_directory(subset="validation", **kw)
    return train, val


def build_model(lr, optimizer_name, loss):
    # Aumento de datos: solo se aplica en entrenamiento.
    augment = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.2),
        layers.RandomTranslation(0.1, 0.1),
        layers.RandomBrightness(0.2),
        layers.RandomContrast(0.2),
    ], name="augment")

    base = VGG16(weights="imagenet", include_top=False, input_shape=IMG_SIZE + (3,))
    base.trainable = False

    inputs = keras.Input(shape=IMG_SIZE + (3,))
    x = augment(inputs)
    x = VGGPreprocess(name="vgg_preprocess")(x)  # RGB -> formato VGG16
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
    ap.add_argument("--fine-tune-epochs", type=int, default=0,
                    help="epocas extra descongelando el bloque 5 de VGG-16 (0 = no hacerlo)")
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

    def callbacks():
        return [
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
            keras.callbacks.ModelCheckpoint(MODEL_PATH, monitor="val_loss", save_best_only=True),
        ]

    model = build_model(args.lr, args.optimizer, args.loss)
    hist = model.fit(train, validation_data=val, epochs=args.epochs,
                     class_weight=class_weight, callbacks=callbacks())
    history = {k: list(v) for k, v in hist.history.items()}

    # Fase 2 (opcional): descongelar el ultimo bloque convolucional de VGG-16 con una tasa de aprendizaje baja.
    if args.fine_tune_epochs > 0:
        base = model.get_layer("vgg16")
        base.trainable = True
        for layer in base.layers:
            layer.trainable = layer.name.startswith("block5")
        model.compile(loss=args.loss, optimizer=keras.optimizers.Adam(1e-5), metrics=["accuracy"])
        start = len(history["loss"])
        hist2 = model.fit(train, validation_data=val, initial_epoch=start,
                          epochs=start + args.fine_tune_epochs,
                          class_weight=class_weight, callbacks=callbacks())
        for k, v in hist2.history.items():
            history[k] += list(v)

    # Evaluacion final con el mejor modelo
    model = keras.models.load_model(MODEL_PATH)
    # Se recorre val una sola vez para que imagenes y etiquetas queden en el mismo orden.
    batches = list(val)
    x_val = np.concatenate([x.numpy() for x, _ in batches])
    y_true = np.concatenate([np.argmax(y.numpy(), axis=1) for _, y in batches])
    y_pred = np.argmax(model.predict(x_val, verbose=0), axis=1)
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))
    print("Matriz de confusion:\n", confusion_matrix(y_true, y_pred))

    CLASSES_PATH.write_text(json.dumps(CLASS_NAMES, ensure_ascii=False))

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for a, k, t in zip(ax, ["loss", "accuracy"], ["Perdida", "Exactitud"]):
        a.plot(history[k], label="entrenamiento")
        a.plot(history["val_" + k], label="validacion")
        a.set_title(t); a.set_xlabel("epoca"); a.legend()
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "curvas_entrenamiento.png", dpi=120)
    print("Modelo guardado en", MODEL_PATH)


if __name__ == "__main__":
    main()
