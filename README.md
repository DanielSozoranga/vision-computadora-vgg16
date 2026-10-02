# Reconocimiento de rostro con VGG-16 (transfer learning)

Actividad calificable de Visión por Computadora: un clasificador de imágenes que decide si una foto
contiene el rostro de **Daniel** o si es **Fondo** (cualquier otra cosa: lugares, objetos u otras personas).
El modelo se presenta en una app de **Streamlit**.

- **Modelo:** VGG-16 preentrenado en ImageNet (base congelada) + capas densas propias, con una fase
  opcional de fine-tuning del último bloque convolucional.
- **Clases:** `Fondo` (0) y `Daniel` (1).
- **Datos:** fotogramas de videos propios (clase `Daniel`) y, para `Fondo`, rostros de otras personas
  (dataset público LFW) y fotos de lugares sin personas (Lorem Picsum).

## Resultados

Pruebas con imágenes que **no** se usaron para entrenar:

| Prueba de la actividad | Entrada | Etiqueta esperada | Resultado |
|---|---|---|---|
| 1 | Fotos nuevas del estudiante | `Daniel` | 56 / 56 |
| 2 | Lugares nuevos | `Fondo` | 10 / 10 |
| 3 | Rostros de celebridades | `Fondo` | 7 / 7 |

En la app de Streamlit, una foto tomada con la webcam (otro cuarto, con lentes) se clasificó como `Daniel`.

### Limitaciones

- **Fotos antiguas, de grupo o capturas de pantalla** (mala calidad, otra época, otra luz) se reconocen peor:
  en un conjunto de 16 fotos de ese tipo el modelo acierta cerca del 40 %. La causa es la poca variedad de
  los datos de entrenamiento (pocas sesiones de fotos del estudiante). Más fotos variadas es la mejora más
  efectiva.
- La precisión de validación (~100 %) es optimista: los fotogramas de un mismo video se parecen mucho. Las
  pruebas con imágenes nuevas de la tabla son la medida real.
- La detección de rostros usa YuNet; si no encuentra un rostro, se analiza la imagen completa.

## Cómo funciona

1. **Detección de rostros** (`src/facecrop.py`): YuNet (incluido en OpenCV) encuentra todos los rostros de la imagen.
2. **Preparación** (`src/facecrop.py`): cada rostro se recorta, se baja a 112 px, se pasa a blanco y negro con
   contraste normalizado y se sube a 224 × 224. Así el modelo aprende la forma de la cara y no la nitidez, el
   color de la cámara ni la ropa. El mismo proceso se usa al entrenar y al predecir.
3. **Clasificación** (`src/inference.py`): se clasifica cada rostro; la imagen es `Daniel` si **alguno** supera
   el umbral (0.5). Si no hay rostros se clasifica la imagen entera.
4. **Entrenamiento** (`src/train.py`): VGG-16 → Flatten → Dense(256, ReLU) → Dropout(0.5) → Dense(2, softmax),
   con aumento de datos (giros, zoom, desplazamiento, brillo, contraste), pesos por clase y EarlyStopping.

## Estructura del proyecto

```
app.py                      App de Streamlit
assets/                     Modelo del detector de rostros (YuNet, .onnx)
src/
  config.py                 Nombre del estudiante, rutas y parámetros
  extract_frames.py         Saca fotogramas de un video
  download_negatives.py     Rostros de otras personas (LFW) para la clase Fondo
  download_scenes.py        Fotos de lugares sin personas para la clase Fondo y para pruebas
  crop_faces.py             Recorta rostros: data/raw -> data/processed
  extract_faces.py          Saca cada rostro de fotos de grupo (para elegirlos a mano)
  train.py                  Entrenamiento
  predict_folder.py         Evalúa una carpeta de imágenes nuevas
  facecrop.py, inference.py, vgg_layer.py   Utilidades
notebooks/                  Notebook original de la actividad
data/raw/                   Fotos originales (no se sube a git)
data/processed/             Rostros recortados (no se sube a git)
models/, reports/           Modelo entrenado y gráficas
```

## Instalación

El proyecto se desarrolló en **WSL (Ubuntu)** con una GPU NVIDIA y Python 3.12.

```bash
uv venv --python 3.12 ~/venvs/vision
source ~/venvs/vision/bin/activate
uv pip install -r requirements.txt "tensorflow[and-cuda]"
```

> TensorFlow aún no soporta Python 3.14, por eso se usa 3.12. Con `opencv-python<5` porque OpenCV 5 ya no
> incluye el detector Haar de respaldo.

Para que TensorFlow encuentre la GPU en WSL, agrega a `~/.bashrc` las rutas de las librerías NVIDIA del venv a
`LD_LIBRARY_PATH` (junto con `/usr/lib/wsl/lib`). Sin GPU el entrenamiento funciona igual, más lento.

## Uso

**1. Datos.** Pon fotos del estudiante en `data/raw/<Nombre>/` (por ejemplo, con
`python src/extract_frames.py video.mp4 --out data/raw/Daniel --every 5`) y llena `data/raw/Fondo/`:

```bash
python src/download_negatives.py --n 1200      # rostros de otras personas
python src/download_scenes.py --n 150 --test 10 # lugares (entrenamiento y prueba)
```

El nombre de la clase se cambia en `src/config.py` (`STUDENT_NAME`).

**2. Preparar y entrenar.**

```bash
python src/crop_faces.py
python src/train.py --epochs 15 --fine-tune-epochs 10
```

**3. Evaluar con imágenes nuevas** (en `pruebas/`, fuera del entrenamiento):

```bash
python src/predict_folder.py pruebas/celebridades --expected Fondo
python src/predict_folder.py pruebas/lugares --expected Fondo
python src/predict_folder.py pruebas/daniel --expected Daniel --save-dir reports/daniel
```

**4. App.**

```bash
streamlit run app.py
```

Permite subir una imagen o usar la webcam, y dibuja un cuadro por rostro (verde = Daniel, rojo = otro) con su
probabilidad.

## Hiperparámetros

`src/train.py` acepta `--epochs`, `--batch-size`, `--optimizer {adam,sgd,rmsprop}`, `--lr`, `--loss` y
`--fine-tune-epochs`. Las curvas de entrenamiento se guardan en `reports/curvas_entrenamiento.png`.

## Notas

- Las fotos del estudiante y el modelo entrenado **no** se suben a git (están en `.gitignore`).
- Dataset LFW: <http://vis-www.cs.umass.edu/lfw/>. Fotos de lugares: <https://picsum.photos> (Unsplash).
- Detector YuNet: <https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet>.
