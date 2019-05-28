#!/bin/bash

export BERT_BASE_DIR=../../bert/pre-trained/uncased_L-12_H-768_A-12
export INIT_FILE=../dat/reddit/model.ckpt-400000
export DATA_FILE=../dat/reddit/proc.tf_record
export OUTPUT_DIR=../output/reddit_embeddings/

#13,6,8 are keto, okcupid, childfree
export SUBREDDITS=13,6,8
export USE_SUB_FLAG=false
export BETA0=1.0
export BETA1=1.0
export GAMMA=1.0

python -m reddit.model.run_causal_bert \
  --seed=0 \
  --do_train=true \
  --do_eval=false \
  --do_predict=true \
  --label_pred=true \
  --unsupervised=true \
  --input_files_or_glob=${DATA_FILE} \
  --vocab_file=${BERT_BASE_DIR}/vocab.txt \
  --bert_config_file=${BERT_BASE_DIR}/bert_config.json \
  --output_dir=${OUTPUT_DIR} \
  --dev_splits=0 \
  --test_splits=0 \
  --max_seq_length=128 \
  --train_batch_size=16 \
  --learning_rate=3e-5 \
  --num_warmup_steps 1000 \
  --num_train_steps=10000 \
  --save_checkpoints_steps=5000 \
  --keep_checkpoints=1 \
  --subreddits=${SUBREDDITS} \
  --beta0=${BETA0} \
  --beta1=${BETA1} \
  --gamma=${GAMMA}
#  --init_checkpoint=${INIT_FILE}