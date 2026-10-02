"""App de Streamlit para la presentacion de la actividad.

Ejecutar:  streamlit run app.py
"""
import sys
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent / "src"))
from config import CLASS_NAMES, IMG_SIZE, MODEL_PATH, STUDENT_NAME  # noqa: E402

st.set_page_config(page_title=f"Clasificador: {STUDENT_NAME} vs Fondo", page_icon="🧑")
st.title(f"¿Eres {STUDENT_NAME}?")
st.caption("VGG-16 con transfer learning · dos clases: rostro del estudiante / fondo")


@st.cache_resource
def get_model():
    from tensorflow import keras
    return keras.models.load_model(MODEL_PATH)


if not MODEL_PATH.exists():
    st.error("No encuentro el modelo. Primero ejecuta: python src/train.py")
    st.stop()

model = get_model()

source = st.radio("Origen de la imagen", ["Subir archivo", "Webcam"], horizontal=True)
file = (st.file_uploader("Imagen", type=["jpg", "jpeg", "png"])
        if source == "Subir archivo" else st.camera_input("Tomar foto"))

if file is not None:
    img = Image.open(file).convert("RGB")
    st.image(img, use_container_width=True)

    # El modelo ya incluye el preprocesado de VGG16: se le pasa RGB en 0-255.
    x = np.asarray(img.resize(IMG_SIZE), dtype="float32")[None]
    probs = model.predict(x, verbose=0)[0]
    best = int(np.argmax(probs))

    if CLASS_NAMES[best] == STUDENT_NAME:
        st.success(f"Etiqueta: **{STUDENT_NAME}** ({probs[best]:.1%})")
    else:
        st.warning(f"Etiqueta: **{CLASS_NAMES[best]}** ({probs[best]:.1%})")
    st.bar_chart({n: float(p) for n, p in zip(CLASS_NAMES, probs)})
