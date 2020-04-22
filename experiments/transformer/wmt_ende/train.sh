#! /usr/bin/env bash

set -e
set -u

if [ $# -lt 1 ]; then
   echo "Usage: ${0} JOB_NAME [FLAGS]"
   exit
fi

programname=${0}
jobname=${1}

shift

ROOT_DIR=/expscratch/${USER}/struct_ensemble_distill
DATA_DIR=${ROOT_DIR}/data/wmt_ende
DATA=${DATA_DIR}/preprocessed
JOB_DIR=${ROOT_DIR}/jobs/${jobname}
mkdir -p ${JOB_DIR}
TRAIN=`realpath ../../../train.py`

python ${TRAIN} -data ${DATA} \
       -log_file ${JOB_DIR}/log.txt \
       -save_model ${JOB_DIR}/model \
       -world_size 1 \
       -gpu_ranks 0 \
       -valid_steps 10000 \
       -early_stopping 4 \
       -early_stopping_criteria accuracy ppl \
       -save_checkpoint_steps 10000 \
       -rnn_size 512 \
       -word_vec_size 512 \
       -batch_type tokens \
       -batch_size 4096 \
       -accum_count 4 \
       -train_steps 200000 \
       -max_generator_batches 2 \
       -normalization tokens \
       -dropout 0.1 \
       -max_grad_norm 0 \
       -optim adam \
       -encoder_type transformer \
       -decoder_type transformer \
       -layers 6 \
       -transformer_ff 2048 \
       -heads 8 \
       -position_encoding \
       -param_init 0 \
       -warmup_steps 8000 \
       -learning_rate 2 \
       -decay_method noam \
       -label_smoothing 0.1 \
       -adam_beta2 0.998 \
       -param_init_glorot


# eof
