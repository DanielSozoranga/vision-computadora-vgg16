"""Saca cada rostro de una carpeta de fotos (grupales, dificiles...) como una imagen aparte.

Sirve para ELEGIR a mano cuales son tuyos:
  - los tuyos       -> los mueves a data/raw/Daniel/
  - los de otros    -> los mueves a data/raw/Fondo/
  - lo que no sea un rostro -> lo borras
crop_faces.py reconoce los archivos que empiezan con "face_" como rostros ya recortados.

Uso:
    python src/extract_faces.py pruebas/dificiles --out pruebas/etiquetar
"""
import argparse
from pathlib import Path

from PIL import Image

from facecrop import crop_box, detect_faces

EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--out", default="pruebas/etiquetar")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    total = 0
    for p in sorted(q for q in Path(args.folder).iterdir() if q.suffix.lower() in EXT):
        img = Image.open(p).convert("RGB")
        boxes = detect_faces(img)
        for k, box in enumerate(boxes):
            crop_box(img, box).save(out / f"face_{p.stem}_{k}.jpg", quality=95)
        total += len(boxes)
        print(f"{p.name}: {len(boxes)} rostro(s)")
    print(f"\n{total} rostros guardados en {out}")


if __name__ == "__main__":
    main()
