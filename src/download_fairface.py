"""Descarga rostros de hombres jovenes latinos (dataset publico FairFace) para la clase Fondo.

Son el tipo de "celebridad parecida al estudiante" que mas confunde al modelo. FairFace trae etiquetas
de etnia, genero y edad, asi que se filtra solo lo que hace falta. Se lee en modo streaming: no baja
el dataset completo, solo lo necesario hasta juntar --n imagenes.

Requiere:  uv pip install datasets
Uso:
    python src/download_fairface.py --n 600
"""
import argparse

from datasets import load_dataset

from config import BACKGROUND_NAME, DATA_DIR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--race", default="Latino_Hispanic")
    ap.add_argument("--gender", default="Male")
    ap.add_argument("--ages", nargs="+", default=["10-19", "20-29", "30-39"])
    ap.add_argument("--split", default="train")
    args = ap.parse_args()

    out = DATA_DIR / BACKGROUND_NAME
    out.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("HuggingFaceM4/FairFace", "1.25", split=args.split, streaming=True)
    race_names = ds.features["race"].names
    gender_names = ds.features["gender"].names
    age_names = ds.features["age"].names
    print("etnias:", race_names)
    print("edades:", age_names)

    saved, seen = 0, 0
    for ex in ds:
        seen += 1
        if (race_names[ex["race"]] == args.race and gender_names[ex["gender"]] == args.gender
                and age_names[ex["age"]] in args.ages):
            ex["image"].convert("RGB").save(out / f"fair_{saved:04d}.jpg", quality=95)
            saved += 1
            if saved % 100 == 0:
                print(f"  {saved} guardadas ({seen} revisadas)")
            if saved >= args.n:
                break
    print(f"\n{saved} rostros guardados en {out} ({seen} revisados)")


if __name__ == "__main__":
    main()
