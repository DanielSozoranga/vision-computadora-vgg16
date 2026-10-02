from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Cambia esto por tu nombre: es la etiqueta 1 de la actividad.
STUDENT_NAME = "Daniel"
BACKGROUND_NAME = "Fondo"

DATA_DIR = ROOT / "data" / "raw"          # fotos originales: data/raw/<STUDENT_NAME>/ y data/raw/Fondo/
PROCESSED_DIR = ROOT / "data" / "processed"   # rostros recortados (los genera crop_faces.py): con esto se entrena
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "modelo_rostro.keras"
CLASSES_PATH = MODEL_DIR / "class_names.json"
REPORT_DIR = ROOT / "reports"

IMG_SIZE = (224, 224)
STUDENT_THRESHOLD = 0.5   # probabilidad minima para decir que un rostro es del estudiante
CLASS_NAMES = [BACKGROUND_NAME, STUDENT_NAME]   # indice 0 = fondo, indice 1 = estudiante
