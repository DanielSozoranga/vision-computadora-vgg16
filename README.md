# Vision por Computadora

Proyecto de la materia de Visión por Computador (clasificación de imágenes, kNN, redes neuronales sobre CIFAR-10, etc.).

## Estructura

- `notebooks/` — notebooks de Jupyter (`.ipynb`) de cada práctica.
- `src/` — código Python reutilizable (funciones, utilidades).
- `data/raw/` — datasets originales (no se sube a git).
- `data/processed/` — datos ya procesados (no se sube a git).
- `.venv/` — entorno virtual de Python (no se sube a git).

## Entorno de desarrollo

Este proyecto usa **WSL (Ubuntu)** + **entorno virtual de Python (venv)**. Ver instrucciones completas de instalación en el chat / documentación del curso.

Para activar el entorno manualmente:

```bash
source .venv/bin/activate
```

Para instalar dependencias:

```bash
pip install -r requirements.txt
```

Para registrar el entorno como kernel de Jupyter:

```bash
python -m ipykernel install --user --name=vision-computador --display-name "Python (vision-computador)"
```
