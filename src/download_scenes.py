"""Descarga fotos de lugares y objetos SIN personas para la clase Fondo.

Fuente: Lorem Picsum (https://picsum.photos), fotos publicas de Unsplash. Las fotos en las que se
detecta un rostro se descartan. Las de entrenamiento y las de prueba salen de semillas distintas,
asi que nunca se repiten.

Uso:
    python src/download_scenes.py --n 150 --test 10
"""
import argparse
import hashlib
import io
import time
import urllib.request
from pathlib import Path

from PIL import Image

from config import BACKGROUND_NAME, DATA_DIR, ROOT
from facecrop import detect_faces

TEST_SEED_OFFSET = 100000   # las fotos de prueba usan semillas que no se cruzan con las de entrenamiento


def fetch(seed):
    url = f"https://picsum.photos/seed/vision{seed}/640/480"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read()


def download(count, first_seed, out_dir, prefix, seen):
    out_dir.mkdir(parents=True, exist_ok=True)
    saved, seed, con_rostro = 0, first_seed, 0
    while saved < count and seed < first_seed + count * 6:
        seed += 1
        try:
            data = fetch(seed)
        except Exception as e:
            print(f"  semilla {seed}: error de descarga ({e})")
            time.sleep(1)
            continue
        digest = hashlib.md5(data).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        img = Image.open(io.BytesIO(data)).convert("RGB")
        if detect_faces(img):
            con_rostro += 1
            continue
        img.save(out_dir / f"{prefix}_{saved:04d}.jpg", quality=95)
        saved += 1
    print(f"{saved} fotos guardadas en {out_dir} ({con_rostro} descartadas por tener rostros)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=150, help="fotos para entrenar (van a data/raw/Fondo)")
    ap.add_argument("--test", type=int, default=10, help="fotos para probar (van a pruebas/lugares)")
    args = ap.parse_args()

    seen = set()
    download(args.n, 0, DATA_DIR / BACKGROUND_NAME, "scene", seen)
    if args.test:
        download(args.test, TEST_SEED_OFFSET, ROOT / "pruebas" / "lugares", "lugar", seen)


if __name__ == "__main__":
    main()
