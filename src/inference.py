"""Clasificacion de una imagen completa: detecta todos los rostros y clasifica cada uno.

Regla: la imagen es del estudiante si ALGUN rostro supera el umbral. Si no hay rostros
(un paisaje, un cuarto...) se clasifica la imagen entera.
"""
import numpy as np
from PIL import ImageDraw

from config import BACKGROUND_NAME, CLASS_NAMES, IMG_SIZE, STUDENT_NAME, STUDENT_THRESHOLD
from facecrop import crop_box, detect_faces, normalize_resolution

STUDENT_IDX = CLASS_NAMES.index(STUDENT_NAME)


def classify(model, img):
    """Devuelve dict con label, confidence, boxes (puede ser []), p_student (por rostro)."""
    img = img.convert("RGB")
    boxes = detect_faces(img)
    if boxes:
        crops = [normalize_resolution(crop_box(img, b), IMG_SIZE) for b in boxes]
    else:
        crops = [normalize_resolution(img, IMG_SIZE)]
    x = np.stack([np.asarray(c, dtype="float32") for c in crops])
    p_student = model.predict(x, verbose=0)[:, STUDENT_IDX]

    best = float(p_student.max())
    if best >= STUDENT_THRESHOLD:
        label, confidence = STUDENT_NAME, best
    else:
        label, confidence = BACKGROUND_NAME, 1.0 - best
    return {"label": label, "confidence": confidence, "boxes": boxes,
            "p_student": [float(p) for p in p_student]}


def annotate(img, result):
    """Dibuja un cuadro por rostro: verde si es el estudiante, rojo si no."""
    out = img.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    width = max(2, out.width // 200)
    for (x, y, w, h), p in zip(result["boxes"], result["p_student"]):
        color = (0, 200, 0) if p >= STUDENT_THRESHOLD else (220, 40, 40)
        draw.rectangle([x, y, x + w, y + h], outline=color, width=width)
        draw.text((x + 3, max(0, y - 12)), f"{STUDENT_NAME} {p:.0%}", fill=color)
    return out
