"""Compara los modelos de los experimentos de hiperparametros con las imagenes de prueba.

Evalua models/modelo_rostro.keras (modelo principal) y todos los models/exp_*.keras en las carpetas de
pruebas/ (imagenes que no se usaron para entrenar) y guarda una tabla en reports/experimentos.md.

Uso:
    python src/compare_experiments.py
"""
from PIL import Image
from tensorflow import keras

import vgg_layer  # noqa: F401  (registra la capa VGGPreprocess)
from config import MODEL_DIR, MODEL_PATH, REPORT_DIR, ROOT, STUDENT_NAME, BACKGROUND_NAME
from inference import classify

EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
# (carpeta dentro de pruebas/, etiqueta esperada, nombre en la tabla)
SETS = [
    ("daniel", STUDENT_NAME, "Tus fotos (video 4)"),
    ("daniel_nuevas", STUDENT_NAME, "Tus fotos (galeria)"),
    ("lugares", BACKGROUND_NAME, "Lugares"),
    ("celebridades", BACKGROUND_NAME, "Celebridades"),
    ("celebridades2", BACKGROUND_NAME, "Parecidos 2"),
    ("celebridades3", BACKGROUND_NAME, "Parecidos 3"),
]


def main():
    sets = []
    for folder, expected, title in SETS:
        path = ROOT / "pruebas" / folder
        files = sorted(p for p in path.iterdir() if p.suffix.lower() in EXT) if path.exists() else []
        if files:
            sets.append((title, expected, [Image.open(p).convert("RGB") for p in files]))

    models = [("principal", MODEL_PATH)] + [(p.stem.removeprefix("exp_"), p)
                                            for p in sorted(MODEL_DIR.glob("exp_*.keras"))]
    rows = []
    for name, path in models:
        if not path.exists():
            continue
        model = keras.models.load_model(path)
        cells, ok_total, n_total = [], 0, 0
        for title, expected, imgs in sets:
            ok = sum(classify(model, im)["label"] == expected for im in imgs)
            cells.append(f"{ok}/{len(imgs)}")
            ok_total, n_total = ok_total + ok, n_total + len(imgs)
        rows.append([name] + cells + [f"{ok_total}/{n_total} ({ok_total / n_total:.0%})"])
        print(name, cells)

    header = ["Modelo"] + [t for t, _, _ in sets] + ["Total"]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    REPORT_DIR.mkdir(exist_ok=True)
    (REPORT_DIR / "experimentos.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines))
    print("\nTabla guardada en reports/experimentos.md")


if __name__ == "__main__":
    main()
