#!/usr/bin/env bash
# Cuenta las imagenes de cada clase (para el video, bloque 2).
cd "$(dirname "$0")/.."
echo "Daniel: $(ls data/processed/Daniel | wc -l)"
echo "Fondo: $(ls data/processed/Fondo | wc -l)"
