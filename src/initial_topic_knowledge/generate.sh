#!/bin/bash

# Example usage: ./generate.sh ../../data/VAST_romanian/vast_train.csv ./from_top_doc top-doc

INPUT_FILE=$1
OUTPUT_DIR=$2
EMBEDDING_TYPE=$3

LOG_FILE=$OUTPUT_DIR/generate.log

rm -rf $OUTPUT_DIR
mkdir $OUTPUT_DIR

for N_SLOTS in 2 4 8 16 32 64 128 256 512
do
  echo -e "Using $N_SLOTS slots..." >> $LOG_FILE
  python init_topic_knowledge.py $EMBEDDING_TYPE $N_SLOTS $INPUT_FILE $OUTPUT_DIR/$N_SLOTS.pt >> $LOG_FILE
done
