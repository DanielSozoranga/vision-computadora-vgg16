# Reconocimiento de rostro con VGG-16 (transfer learning)

Actividad calificable de Visión por Computadora: un clasificador de imágenes que decide si una foto
contiene el rostro de **Daniel** o si es **Fondo** (cualquier otra cosa: lugares, objetos u otras personas).
El modelo se presenta en una app de **Streamlit**.

- **Modelo:** VGG-16 preentrenado en ImageNet (base congelada) + capas densas propias, con una fase
  opcional de fine-tuning del último bloque convolucional.
- **Clases:** `Fondo` (0) y `Daniel` (1).
- **Datos:**
  - `Daniel`: fotogramas de videos propios (celular y webcam) y fotos de galería.
  - `Fondo`: rostros de otras personas (LFW y FairFace), fotos de lugares sin personas (Lorem Picsum) y
    rostros "difíciles" encontrados con *hard negative mining* (ver más abajo).

## Resultados

Pruebas con imágenes que **no** se usaron para entrenar:

| Prueba de la actividad | Entrada | Etiqueta esperada | Resultado |
|---|---|---|---|
| 1 | Fotos nuevas del estudiante (video aparte) | `Daniel` | 56 / 56 |
| 1 | Fotos nuevas de galería (otros lugares, gorra, perfil) | `Daniel` | 8 / 8 |
| 1 | Fotos con la webcam en la app (de noche, de lado, con lentes, con la mano levantada...) | `Daniel` | 7 / 9 (fallan dos muecas exageradas) |
| 1 | 16 fotos difíciles (grupales, antiguas, capturas de pantalla; nunca usadas para entrenar) | `Daniel` | 8 / 16 |
| 2 | Lugares nuevos | `Fondo` | 10 / 10 |
| 3 | Rostros de celebridades | `Fondo` | 7 / 7 |
| 3 | Celebridades parecidas al estudiante | `Fondo` | 8 / 9 |

### Limitaciones

- **Personas muy parecidas al estudiante:** una foto de una persona de rasgos similares se clasifica como
  `Daniel` con alta probabilidad. Es el error más difícil de eliminar con una sola persona como clase.
- **Fotos antiguas, de grupo o capturas de pantalla** (mala calidad, otra época, otra luz) se reconocen
  solo en parte (8 de 16 en el conjunto de prueba difícil). La causa es que el entrenamiento solo incluye
  al estudiante en las sesiones de fotos recientes; esas 16 fotos no se usaron para entrenar.
- **Poses extremas** (mano cubriendo media cara, muecas exageradas) pueden dar `Fondo`.
- La precisión de validación (~99-100 %) es optimista: los fotogramas de un mismo video se parecen mucho.
  Las pruebas con imágenes nuevas de la tabla son la medida real.

## Cómo funciona

1. **Detección de rostros** (`src/facecrop.py`): YuNet (incluido en OpenCV) encuentra todos los rostros de la imagen.
2. **Preparación** (`src/facecrop.py`): cada rostro se recorta ajustado a la cara, se baja a 112 px, se pasa a
   blanco y negro con ecualización local de contraste (CLAHE) y se sube a 224 × 224. Así el modelo aprende la
   forma de la cara y no la nitidez, el color de la cámara, la ropa ni el contraluz. El mismo proceso se usa
   al entrenar y al predecir.
3. **Clasificación** (`src/inference.py`): se clasifica cada rostro; la imagen es `Daniel` si **alguno** supera
   el umbral (0.8). Si no hay rostros se clasifica la imagen entera.
4. **Entrenamiento** (`src/train.py`): VGG-16 → Flatten → Dense(256, ReLU) → Dropout(0.5) → Dense(2, softmax),
   con aumento de datos (giros, zoom, desplazamiento, brillo, contraste), pesos por clase, EarlyStopping y
   una fase final de fine-tuning del bloque 5.
5. **Calidad de imagen**: de cada imagen se guarda además una versión degradada (pequeña y comprimida en JPEG)
   en las dos clases, para que el modelo no aprenda "borroso = otra persona".
6. **Hard negative mining** (`src/mine_negatives.py`): el modelo ya entrenado revisa miles de rostros públicos
   (LFW y hombres jóvenes latinos de FairFace) y los que confunde con el estudiante se agregan a `Fondo`
   para el siguiente entrenamiento. Nunca se usan las imágenes de prueba.

## Estructura del proyecto

```
app.py                      App de Streamlit
assets/                     Modelo del detector de rostros (YuNet, .onnx)
src/
  config.py                 Nombre del estudiante, rutas, umbral y parámetros
  extract_frames.py         Saca fotogramas de un video
  download_negatives.py     Rostros de otras personas (LFW) para la clase Fondo
  download_fairface.py      Rostros de hombres jóvenes latinos (FairFace) para la clase Fondo
  download_scenes.py        Fotos de lugares sin personas para la clase Fondo y para pruebas
  mine_negatives.py         Búsqueda de rostros que el modelo confunde con el estudiante
  crop_faces.py             Recorta rostros: data/raw -> data/processed
  extract_faces.py          Saca cada rostro de fotos de grupo (para elegirlos a mano)
  train.py                  Entrenamiento
  predict_folder.py         Evalúa una carpeta de imágenes nuevas
  compare_experiments.py    Compara los modelos de los experimentos de hiperparámetros
run_experiments.sh          Entrena los 7 experimentos de hiperparámetros
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
> incluye el detector Haar de respaldo. Para `download_fairface.py` y `mine_negatives.py --source fairface`
> hace falta `uv pip install datasets`.

Para que TensorFlow encuentre la GPU en WSL, agrega a `~/.bashrc` las rutas de las librerías NVIDIA del venv a
`LD_LIBRARY_PATH` (junto con `/usr/lib/wsl/lib`). Sin GPU el entrenamiento funciona igual, más lento.

## Uso

**1. Datos.** Pon fotos del estudiante en `data/raw/<Nombre>/` (por ejemplo, con
`python src/extract_frames.py video.mp4 --out data/raw/Daniel --every 6`) y llena `data/raw/Fondo/`:

```bash
python src/download_negatives.py --n 1200       # rostros de otras personas (LFW)
python src/download_scenes.py --n 150 --test 10 # lugares (entrenamiento y prueba)
python src/download_fairface.py --n 600         # hombres jóvenes latinos (FairFace)
```

El nombre de la clase se cambia en `src/config.py` (`STUDENT_NAME`).

**2. Preparar y entrenar.**

```bash
python src/crop_faces.py
python src/train.py --epochs 15 --fine-tune-epochs 10
```

**3. (Opcional) Hard negative mining y reentrenar.**

```bash
python src/mine_negatives.py --source fairface --pool 4000 --skip 600 --threshold 0.05
python src/crop_faces.py && python src/train.py --epochs 15 --fine-tune-epochs 10
```

**4. Evaluar con imágenes nuevas** (en `pruebas/`, fuera del entrenamiento):

```bash
python src/predict_folder.py pruebas/celebridades --expected Fondo
python src/predict_folder.py pruebas/lugares --expected Fondo
python src/predict_folder.py pruebas/daniel --expected Daniel --save-dir reports/daniel
```

**5. App.**

```bash
streamlit run app.py
```

Abre `http://localhost:8501` (con `localhost`, el navegador permite usar la webcam). Permite subir una imagen o
tomar una foto, y dibuja un cuadro por rostro (verde = Daniel, rojo = otro) con su probabilidad.

## Hiperparámetros

`src/train.py` acepta `--epochs`, `--batch-size`, `--optimizer {adam,sgd,rmsprop}`, `--lr`, `--loss`,
`--no-augment` y `--fine-tune-epochs`. Las curvas de entrenamiento se guardan en
`reports/curvas_entrenamiento.png`.

### Experimentos de hiperparámetros

`run_experiments.sh` entrena 7 modelos cortos (10 épocas, sin fine-tuning), cambiando **una sola cosa** respecto
a la referencia (Adam, lr 1e-4, lote 16, `categorical_crossentropy`, con aumento de datos).
`src/compare_experiments.py` los evalúa con imágenes que no se usaron para entrenar:

| Modelo | Qué cambia | Tus fotos (video) | Tus fotos (galería) | Lugares | Celebridades | Parecidos 2 | Parecidos 3 | Total |
|---|---|---|---|---|---|---|---|---|
| **principal** | 15 épocas + 10 de fine-tuning (entrenamiento completo) | 56/56 | 8/8 | 10/10 | 7/7 | 3/3 | 5/6 | **89/90 (99%)** |
| rmsprop | Optimizador RMSprop | 56/56 | 8/8 | 10/10 | 6/7 | 2/3 | 5/6 | 87/90 (97%) |
| lote32 | Tamaño de lote 32 | 56/56 | 8/8 | 10/10 | 5/7 | 2/3 | 5/6 | 86/90 (96%) |
| epocas3 | Solo 3 épocas | 56/56 | 8/8 | 10/10 | 5/7 | 2/3 | 4/6 | 85/90 (94%) |
| sin_aumento | Sin aumento de datos | 55/56 | 5/8 | 10/10 | 7/7 | 2/3 | 6/6 | 85/90 (94%) |
| sgd | Optimizador SGD (lr 1e-3) | 56/56 | 8/8 | 10/10 | 4/7 | 1/3 | 5/6 | 84/90 (93%) |
| referencia | Adam, lote 16, 10 épocas | 56/56 | 4/8 | 10/10 | 5/7 | 2/3 | 4/6 | 81/90 (90%) |
| perdida_mse | Pérdida de error cuadrático medio | 56/56 | 8/8 | 10/10 | 3/7 | 0/3 | 2/6 | 79/90 (88%) |

Conclusiones:

- **La función de pérdida es lo que más importa:** `mean_squared_error` es claramente la peor (79/90), sobre todo
  con rostros de otras personas. `categorical_crossentropy` es la adecuada para clasificación con softmax.
- **Optimizador:** en validación, Adam fue mejor (98.1 % contra 96.7 % de RMSprop y 95.3 % de SGD), pero con
  imágenes nuevas RMSprop quedó por encima (87/90 contra 81/90). La diferencia viene de solo 4 fotos de galería,
  así que **no es concluyente**: con tan pocas imágenes de prueba, 4 o 5 aciertos de diferencia pueden ser ruido.
- **Épocas:** 3 épocas (85/90) quedan por debajo de 10 (81 a 87/90) y del entrenamiento completo con fine-tuning
  (89/90), que es el mejor modelo.
- **Aumento de datos:** sin aumento la exactitud de validación sube a 99.5 %, pero eso es memorización de
  fotogramas muy parecidos; con imágenes nuevas el resultado no mejora (85/90).
- **Tamaño de lote:** 16 y 32 dan resultados parecidos.
- El modelo **principal** es el mejor no por un hiperparámetro aislado, sino por el protocolo completo
  (más épocas, fine-tuning del bloque 5 y búsqueda de rostros difíciles).

Las conclusiones se basan en conjuntos de prueba pequeños (90 imágenes en total) y una sola corrida por
configuración; son orientativas.

## Notas

- Las fotos del estudiante y el modelo entrenado **no** se suben a git (están en `.gitignore`).
- Dataset LFW: <http://vis-www.cs.umass.edu/lfw/>. FairFace: <https://github.com/joojs/fairface>.
  Fotos de lugares: <https://picsum.photos> (Unsplash).
- Detector YuNet: <https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet>.
