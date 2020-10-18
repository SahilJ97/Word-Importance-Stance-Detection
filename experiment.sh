#!/bin/bash

OUTPUT_DIR=./training_output
LOG_FILE=$OUTPUT_DIR/train.log

rm -rf $OUTPUT_DIR
mkdir $OUTPUT_DIR

python src/model_configs/generate_model_configs.py

for f in $"ls -d src/model_configs/configs/*.jsonnet"
do
  f_base=$(basename $f .jsonnet)
  echo "Training $f_base" >> $LOG_FILE
  allennlp train "$f" -s training_output/"$f_base" \
  --include-package src.classifiers --include-package allennlp.models --include-package src.vast_reader -f
  allennlp evaluate "$f_base"/model.tar.gz data/VAST_romanian/vast_test.csv \
  --include-package src.classifiers --include-package allennlp.models --include-package src.vast_reader
done