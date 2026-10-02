"""Extrae fotogramas de un video corto tuyo (grabado con el celular o la camara de Windows).

Uso:
    python src/extract_frames.py mi_video.mp4 --out data/raw/Estudiante --every 5
"""
import argparse
from pathlib import Path

import cv2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--out", required=True)
    ap.add_argument("--every", type=int, default=5, help="guardar 1 de cada N fotogramas")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(args.video)
    i = saved = 0
    start = len(list(out.glob("*.jpg")))
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % args.every == 0:
            cv2.imwrite(str(out / f"{Path(args.video).stem}_{start + saved:04d}.jpg"), frame)
            saved += 1
        i += 1
    cap.release()
    print(f"{saved} imagenes guardadas en {out}")


if __name__ == "__main__":
    main()
