#!/usr/bin/env bash
# Copia dos imagenes de prueba (que nunca se usaron para entrenar) a Descargas\demo_video
# para subirlas en la demostracion de la app.
cd "$(dirname "$0")/.."
dest="/mnt/c/Users/danie/Downloads/demo_video"
mkdir -p "$dest"
cp pruebas/lugares/lugar_0000.jpg "$dest/lugar.jpg"
celeb=$(ls pruebas/celebridades | head -1)
cp "pruebas/celebridades/$celeb" "$dest/celebridad.${celeb##*.}"
echo "Listo. Archivos en $dest:"
ls "$dest"
