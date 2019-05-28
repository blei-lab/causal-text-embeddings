#!/bin/bash

export BERT_BASE_DIR=../../bert/pre-trained/uncased_L-12_H-768_A-12

export DATA_FILE=../dat/reddit/proc.tf_record
export OUTPUT_DIR=../output/reddit_embeddings/

#rm -rf $OUTPUT_DIR
python -m model.run_unsupervised_pretraining \
  --seed=0 \
  --do_train=true \
  --input_file=${DATA_FILE} \
  --vocab_file=${BERT_BASE_DIR}/vocab.txt \
  --bert_config_file=${BERT_BASE_DIR}/bert_config.json \
  --output_dir=${OUTPUT_DIR} \
  --max_seq_length=256 \
  --train_batch_size=16 \
  --learning_rate=3e-5 \
  --num_warmup_steps 200 \
  --num_train_steps=175000 \
  --save_checkpoints_steps 5000 \
  --keep_checkpoints 3