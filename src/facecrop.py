"""Deteccion y recorte de rostros con OpenCV.

Detector principal: YuNet (red neuronal chica incluida en OpenCV; ve rostros de lado, con gorra,
con lentes o chicos). Si no esta el archivo del modelo en assets/, se usa Haar como respaldo.

Se recorta el rostro para que el modelo aprenda la CARA y no la ropa, la pared o la luz del cuarto.
Lo usan crop_faces.py (datos de entrenamiento), predict_folder.py y app.py (prediccion).
"""
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

YUNET_PATH = Path(__file__).resolve().parent.parent / "assets" / "face_detection_yunet_2023mar.onnx"

_haar = None
_yunet = None
_yunet_checked = False


def _get_yunet():
    global _yunet, _yunet_checked
    if not _yunet_checked:
        _yunet_checked = True
        if YUNET_PATH.exists():
            _yunet = cv2.FaceDetectorYN.create(str(YUNET_PATH), "", (320, 320), 0.6, 0.3, 5000)
        else:
            print(f"AVISO: no existe {YUNET_PATH}; se usa el detector Haar (menos preciso).")
    return _yunet


def _get_haar():
    global _haar
    if _haar is None:
        _haar = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    return _haar


def detect_faces(img, max_faces=8):
    """Todos los rostros de la imagen como cajas (x, y, w, h), del mas grande al mas chico."""
    img = img.convert("RGB")
    W, H = img.size
    det = _get_yunet()
    if det is not None:
        bgr = np.ascontiguousarray(np.asarray(img)[:, :, ::-1])
        det.setInputSize((W, H))
        _, found = det.detect(bgr)
        boxes = []
        if found is not None:
            for f in found:
                x, y, w, h = (int(round(v)) for v in f[:4])
                x, y = max(0, x), max(0, y)
                w, h = min(w, W - x), min(h, H - y)
                if min(w, h) >= 30:
                    boxes.append((x, y, w, h))
    else:
        gray = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2GRAY)
        min_size = max(40, int(0.04 * min(W, H)))
        found = _get_haar().detectMultiScale(gray, scaleFactor=1.08, minNeighbors=4,
                                             minSize=(min_size, min_size))
        boxes = [tuple(int(v) for v in f) for f in found]
    boxes.sort(key=lambda b: -b[2] * b[3])
    return boxes[:max_faces]


def crop_box(img, box, margin=0.0):
    """Recorte cuadrado alrededor de la caja del rostro (con un margen)."""
    x, y, w, h = box
    cx, cy = x + w / 2, y + h / 2
    half = max(w, h) * (1 + 2 * margin) / 2
    return img.convert("RGB").crop((int(cx - half), int(cy - half), int(cx + half), int(cy + half)))


def crop_largest_face(img, margin=0.0):
    """Devuelve el rostro mas grande como imagen PIL cuadrada, o None si no hay rostro."""
    boxes = detect_faces(img)
    return crop_box(img, boxes[0], margin) if boxes else None


LOW_RES = 112


def normalize_resolution(img, size):
    """Baja la imagen a LOW_RES px y la sube a `size`.

    Asi todas las imagenes (tus fotos nitidas del celular y los rostros chicos de otras personas)
    tienen la misma nitidez y el modelo no puede distinguirlas por la calidad de la imagen.
    """
    small = ImageOps.fit(img.convert("RGB"), (LOW_RES, LOW_RES), Image.LANCZOS)
    # Blanco y negro + contraste normalizado: quita el color de la camara y de la luz del lugar,
    # para que el modelo aprenda la forma de la cara y no "en que cuarto se tomo la foto".
    # CLAHE: iguala el contraste por zonas, asi un rostro a contraluz (ventana detras) o con media cara
    # en sombra queda parecido a uno bien iluminado.
    gray = np.asarray(ImageOps.grayscale(small))
    gray = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4)).apply(gray)
    small = Image.fromarray(gray).convert("RGB")
    return small.resize(size, Image.BICUBIC)
