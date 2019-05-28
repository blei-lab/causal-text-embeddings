#!/bin/bash

BERT_BASE_DIR=../../bert/pre-trained/uncased_L-12_H-768_A-12
DATA_FILE=../dat/PeerRead/proc/arxiv-all.tf_record
OUTPUT_DIR=../output/PeerRead/local_test
#INIT_DIR=../../output/unsupervised_PeerRead_embeddings/
#INIT_FILE=$INIT_DIR/model.ckpt-175000


#rm -rf $OUTPUT_DIR

python -m PeerRead.model.run_causal_bert \
  --seed=0 \
  --do_train=true \
  --do_eval=false \
  --do_predict=true \
  --input_files_or_glob=$DATA_FILE \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --max_seq_length=250 \
  --output_dir=$OUTPUT_DIR \
  --train_batch_size=16 \
  --learning_rate=3e-5 \
  --num_warmup_steps 200 \
  --num_train_steps=4500 \
  --save_checkpoint_steps=3000 \
  --unsupervised=True \
  --label_pred=True \
  --num_splits=10 \
  --test_splits=0 \
  --dev_splits=0 \
  --simulated='real' \
  --treatment='buzzy_title'
#  --init_checkpoint=${INIT_FILE}
