#!/usr/bin/env bash
RAW='data/raw'
INTERIM='data/interim'
PROCESSED='data/processed'

for subset in 'train' 'test'; do
    python3 src/parse_str_cols.py ${RAW}/${subset}.csv ${INTERIM}/${subset}_parsed.csv
    python3 src/fill_nans.py ${INTERIM}/${subset}_parsed.csv ${INTERIM}/${subset}_filled_nans.csv
    python3 src/data_to_arrays.py ${INTERIM}/${subset}_filled_nans.csv ${PROCESSED}/${subset}.npy
done;
