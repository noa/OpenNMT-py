#! /usr/bin/env bash

set -e
set -u

PREPROCESS=`realpath ../../../preprocess.py`

DATA_DIR=/expscratch/${USER}/struct_ensemble_distill/data/wmt_ende
OUTPUT_DIR=${DATA_DIR}/preprocessed

TRAIN_EN=${DATA_DIR}/train.en
TRAIN_DE=${DATA_DIR}/train.de
VALID_EN=${DATA_DIR}/valid.en
VALID_DE=${DATA_DIR}/valid.de

SRC_LEN=100
TGT_LEN=100

python ${PREPROCESS} \
       -train_src ${TRAIN_EN}  \
       -train_tgt ${TRAIN_DE} \
       -valid_src ${VALID_EN} \
       -valid_tgt ${VALID_DE} \
       -save_data ${OUTPUT_DIR} \
       -src_seq_length ${SRC_LEN} \
       -tgt_seq_length ${TGT_LEN} \
       -share_vocab

# eof
