"""App de Streamlit para la presentacion de la actividad.

Ejecutar:  streamlit run app.py
"""
import sys
from pathlib import Path

import streamlit as st
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent / "src"))
from config import BACKGROUND_NAME, MODEL_PATH, STUDENT_NAME  # noqa: E402
from inference import annotate, classify  # noqa: E402
import vgg_layer  # noqa: E402,F401  (registra la capa VGGPreprocess para load_model)

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

    # Se detectan todos los rostros y se clasifica cada uno; basta que uno sea el del estudiante.
    result = classify(model, img)
    caption = (f"{len(result['boxes'])} rostro(s) detectado(s)" if result["boxes"]
               else "No se detectó rostro: se analizó la imagen completa")
    st.image(annotate(img, result), caption=caption, width="stretch")

    if result["label"] == STUDENT_NAME:
        st.success(f"Etiqueta: **{STUDENT_NAME}** ({result['confidence']:.1%})")
    else:
        st.warning(f"Etiqueta: **{result['label']}** ({result['confidence']:.1%})")
    st.bar_chart({STUDENT_NAME: max(result["p_student"]), BACKGROUND_NAME: 1 - max(result["p_student"])})
