"""Descarga rostros de OTRAS personas (dataset publico LFW) para la clase Fondo.

Uso:
    python src/download_negatives.py --n 300
"""
import argparse
import random

import cv2
import numpy as np
from sklearn.datasets import fetch_lfw_people

from config import BACKGROUND_NAME, DATA_DIR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300, help="cuantas imagenes guardar")
    ap.add_argument("--per-person", type=int, default=2, help="maximo de fotos por persona")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out = DATA_DIR / BACKGROUND_NAME
    out.mkdir(parents=True, exist_ok=True)

    print("Descargando LFW (la primera vez son ~200 MB)...")
    # slice_ completo: la imagen entera de 250x250. Por defecto sklearn recorta a 94x125 px
    # (muy borroso al agrandarlo) y el modelo aprenderia "borroso = otra persona".
    lfw = fetch_lfw_people(color=True, resize=1.0, funneled=True, download_if_missing=True,
                           slice_=(slice(0, 250), slice(0, 250)))

    rng = random.Random(args.seed)
    idx = list(range(len(lfw.images)))
    rng.shuffle(idx)

    per_person, saved = {}, 0
    for i in idx:
        who = int(lfw.target[i])
        if per_person.get(who, 0) >= args.per_person:
            continue
        img = lfw.images[i]
        if img.max() <= 1.0:          # sklearn entrega 0-1 en algunas versiones
            img = img * 255.0
        img = np.clip(img, 0, 255).astype("uint8")
        cv2.imwrite(str(out / f"lfw_{saved:04d}.jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        per_person[who] = per_person.get(who, 0) + 1
        saved += 1
        if saved >= args.n:
            break
    print(f"{saved} rostros de otras personas guardados en {out}")


if __name__ == "__main__":
    main()
