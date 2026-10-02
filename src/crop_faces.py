"""Genera data/processed/<clase>/ a partir de data/raw/<clase>/ recortando el rostro.

- Clase del estudiante: solo se queda con las fotos donde se detecta un rostro.
- Clase Fondo: si hay rostro (otras personas) se recorta; si no (lugares, objetos) se queda la imagen completa.
- De cada imagen se guardan DOS versiones: la original y otra "de mala calidad" (mas chica y comprimida
  en JPEG, como las fotos de WhatsApp o capturas de redes). Se hace igual en las dos clases para que
  el modelo no aprenda "borroso = otra persona".

Uso:
    python src/crop_faces.py
"""
import io
import random
import shutil

from PIL import Image, ImageFilter

from config import BACKGROUND_NAME, DATA_DIR, IMG_SIZE, PROCESSED_DIR, STUDENT_NAME
from facecrop import crop_largest_face, normalize_resolution

EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
REPEAT_HAND_PICKED = 6   # veces que se repite cada rostro elegido a mano (archivos face_*.jpg)


def degrade(img, rng):
    """Version de mala calidad: reduce a 40-100 px, comprime en JPEG y a veces desenfoca un poco."""
    side = rng.randint(40, 100)
    small = img.convert("RGB").resize((side, side), Image.LANCZOS)
    if rng.random() < 0.4:
        small = small.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 1.0)))
    buf = io.BytesIO()
    small.save(buf, "JPEG", quality=rng.randint(25, 85))
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def main():
    rng = random.Random(42)
    if PROCESSED_DIR.exists():
        for sub in PROCESSED_DIR.iterdir():
            if sub.is_dir():
                shutil.rmtree(sub)

    for cls in (BACKGROUND_NAME, STUDENT_NAME):
        src, dst = DATA_DIR / cls, PROCESSED_DIR / cls
        dst.mkdir(parents=True, exist_ok=True)
        files = sorted(p for p in src.iterdir() if p.suffix.lower() in EXT)
        faces = whole = skipped = 0
        for p in files:
            try:
                img = Image.open(p).convert("RGB")
            except Exception:
                skipped += 1
                continue
            if p.name.startswith("face_"):
                # Rostro ya recortado a mano con extract_faces.py: se usa tal cual y se repite.
                out, faces, n_versions = img, faces + 1, REPEAT_HAND_PICKED
            else:
                face = crop_largest_face(img)
                n_versions = 2
                if face is not None:
                    out, faces = face, faces + 1
                elif cls == BACKGROUND_NAME:
                    out, whole = img, whole + 1
                else:
                    skipped += 1
                    continue
            for r in range(n_versions):
                version = out if r % 2 == 0 else degrade(out, rng)
                normalize_resolution(version, IMG_SIZE).save(dst / f"{p.stem}_{r}.jpg", quality=95)
        print(f"{cls}: {len(files)} originales -> {faces} rostros recortados, "
              f"{whole} imagenes completas, {skipped} descartadas")


if __name__ == "__main__":
    main()
