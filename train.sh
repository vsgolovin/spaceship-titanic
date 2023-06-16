#!/usr/bin/env bash

INP_DIR="data/processed"
MODEL="xgbc"

python3 src/train_model.py \
    ${INP_DIR}/train.npy \
    ${INP_DIR}/labels.npy \
    --model-type $MODEL \
    --save-to models/${MODEL}.pkl \
    --no-val