"""Prueba el modelo con una carpeta de imagenes NUEVAS (que no se usaron para entrenar).

Uso:
    python src/predict_folder.py pruebas/daniel --expected Daniel
    python src/predict_folder.py pruebas/dificiles --expected Daniel --save-dir reports/dificiles
"""
import argparse
from pathlib import Path

from PIL import Image
from tensorflow import keras

import vgg_layer  # noqa: F401  (registra la capa VGGPreprocess)
from config import CLASS_NAMES, MODEL_PATH
from inference import annotate, classify

EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--expected", required=True, choices=CLASS_NAMES)
    ap.add_argument("--save-dir", help="guarda cada imagen con los rostros marcados")
    args = ap.parse_args()

    files = sorted(p for p in Path(args.folder).iterdir() if p.suffix.lower() in EXT)
    if not files:
        raise SystemExit(f"No hay imagenes en {args.folder}")
    if args.save_dir:
        Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    model = keras.models.load_model(MODEL_PATH)
    ok, fails, no_face = 0, [], 0
    for p in files:
        img = Image.open(p).convert("RGB")
        res = classify(model, img)
        no_face += not res["boxes"]
        if res["label"] == args.expected:
            ok += 1
        else:
            fails.append((p.name, res))
        if args.save_dir:
            annotate(img, res).save(Path(args.save_dir) / f"{p.stem}.jpg")

    print(f"\nEsperado: {args.expected} | Aciertos: {ok}/{len(files)} ({ok / len(files):.0%})")
    if no_face:
        print(f"  (sin rostro detectado, se uso la imagen completa: {no_face})")
    for name, res in fails:
        caras = ", ".join(f"{p:.0%}" for p in res["p_student"])
        print(f"  FALLO {name}: dijo {res['label']} | rostros: {len(res['boxes'])} | prob. de ser tu: {caras}")


if __name__ == "__main__":
    main()
