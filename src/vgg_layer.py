"""Capa de preprocesado de VGG-16 que Keras puede guardar y volver a cargar.

Hay que importar este modulo antes de load_model() (train.py y app.py ya lo hacen).
"""
import keras
from keras.applications.vgg16 import preprocess_input


@keras.saving.register_keras_serializable(package="vision")
class VGGPreprocess(keras.layers.Layer):
    """RGB 0-255 -> formato que espera VGG-16 (BGR y resta de la media de ImageNet)."""

    def call(self, x):
        return preprocess_input(x)

    def compute_output_shape(self, input_shape):
        return input_shape
