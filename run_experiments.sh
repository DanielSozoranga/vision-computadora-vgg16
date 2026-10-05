#!/usr/bin/env bash
# Experimentos de hiperparametros: cada corrida cambia UNA cosa respecto a la referencia
# (Adam, lr 1e-4, lote 16, categorical_crossentropy, con aumento de datos, 10 epocas, sin fine-tuning).
# Cada modelo se guarda como models/exp_<nombre>.keras; el modelo principal NO se toca.
#
# Uso:   bash run_experiments.sh        (tarda ~1 hora)
#        python src/compare_experiments.py
set -e
cd "$(dirname "$0")"

COMMON="--epochs 10 --fine-tune-epochs 0"

python src/train.py --tag referencia      $COMMON
python src/train.py --tag sgd             $COMMON --optimizer sgd --lr 1e-3
python src/train.py --tag rmsprop         $COMMON --optimizer rmsprop
python src/train.py --tag lote32          $COMMON --batch-size 32
python src/train.py --tag perdida_mse     $COMMON --loss mean_squared_error
python src/train.py --tag sin_aumento     $COMMON --no-augment
python src/train.py --tag epocas3         --epochs 3 --fine-tune-epochs 0
