"""Hard negative mining: busca rostros de OTRAS personas que el modelo confunde con el estudiante
y los agrega a la clase Fondo para reentrenar.

Fuentes publicas: LFW (figuras publicas) o FairFace (filtrado a hombres jovenes latinos, el tipo de
persona que mas se parece a la clase del estudiante). Nunca se usan las imagenes de prueba.
Hay que tener un modelo ya entrenado.

Uso:
    python src/mine_negatives.py --source lfw --threshold 0.05 --max 600
    python src/mine_negatives.py --source fairface --pool 4000 --skip 600 --threshold 0.05
    python src/crop_faces.py && python src/train.py --epochs 15 --fine-tune-epochs 10
"""
import argparse
import os
import sys

import numpy as np
from PIL import Image
from tensorflow import keras

import vgg_layer  # noqa: F401  (registra la capa VGGPreprocess)
from config import BACKGROUND_NAME, CLASS_NAMES, DATA_DIR, IMG_SIZE, MODEL_PATH, STUDENT_NAME
from facecrop import crop_box, detect_faces, normalize_resolution

STUDENT_IDX = CLASS_NAMES.index(STUDENT_NAME)


def lfw_images(per_person):
    from sklearn.datasets import fetch_lfw_people
    lfw = fetch_lfw_people(color=True, resize=1.0, funneled=True, download_if_missing=True,
                           slice_=(slice(0, 250), slice(0, 250)))
    count = {}
    for i, arr in enumerate(lfw.images):
        who = int(lfw.target[i])
        if count.get(who, 0) >= per_person:
            continue
        count[who] = count.get(who, 0) + 1
        a = arr * 255.0 if arr.max() <= 1.0 else arr
        yield Image.fromarray(np.clip(a, 0, 255).astype("uint8"))


def fairface_images(pool, skip, race="Latino_Hispanic", gender="Male", ages=("10-19", "20-29", "30-39")):
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceM4/FairFace", "1.25", split="train", streaming=True)
    race_names, gender_names, age_names = (ds.features[k].names for k in ("race", "gender", "age"))
    matched = given = 0
    for ex in ds:
        if (race_names[ex["race"]] == race and gender_names[ex["gender"]] == gender
                and age_names[ex["age"]] in ages):
            matched += 1
            if matched <= skip:          # las primeras ya se descargaron con download_fairface.py
                continue
            yield ex["image"].convert("RGB")
            given += 1
            if given >= pool:
                return


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["lfw", "fairface"], default="lfw")
    ap.add_argument("--threshold", type=float, default=0.05,
                    help="se guardan los rostros con probabilidad de ser el estudiante >= este valor")
    ap.add_argument("--max", type=int, default=600, help="maximo de rostros a agregar")
    ap.add_argument("--per-person", type=int, default=4, help="(lfw) maximo de fotos por persona")
    ap.add_argument("--pool", type=int, default=4000, help="(fairface) cuantas imagenes revisar")
    ap.add_argument("--skip", type=int, default=600, help="(fairface) saltar las primeras N coincidencias")
    args = ap.parse_args()

    out = DATA_DIR / BACKGROUND_NAME
    out.mkdir(parents=True, exist_ok=True)
    model = keras.models.load_model(MODEL_PATH)

    print(f"Cargando {args.source}...")
    source = lfw_images(args.per_person) if args.source == "lfw" else fairface_images(args.pool, args.skip)

    keep, imgs = [], []
    for n, img in enumerate(source, 1):
        boxes = detect_faces(img)
        if boxes:
            keep.append(img)
            imgs.append(normalize_resolution(crop_box(img, boxes[0]), IMG_SIZE))
        if n % 500 == 0:
            print(f"  {n} imagenes revisadas ({len(imgs)} con rostro)")
    print(f"{len(imgs)} rostros analizados")

    probs = []
    for s in range(0, len(imgs), 128):
        batch = np.stack([np.asarray(im, dtype="float32") for im in imgs[s:s + 128]])
        probs.append(model.predict(batch, verbose=0)[:, STUDENT_IDX])
    probs = np.concatenate(probs)

    for t in (0.5, 0.2, 0.05, 0.01):
        print(f"  confundidos con {STUDENT_NAME} (prob >= {t}): {(probs >= t).sum()}")

    order = [i for i in np.argsort(-probs) if probs[i] >= args.threshold][:args.max]
    first = len(list(out.glob("hard_*.jpg")))     # sigue la numeracion; no pisa rondas anteriores
    for n, i in enumerate(order):
        keep[i].save(out / f"hard_{first + n:04d}.jpg", quality=95)
    print(f"\n{len(order)} rostros dificiles agregados a {out}")

    sys.stdout.flush()
    os._exit(0)     # evita el 'Aborted' que da la libreria de datasets al cerrar


if __name__ == "__main__":
    main()
