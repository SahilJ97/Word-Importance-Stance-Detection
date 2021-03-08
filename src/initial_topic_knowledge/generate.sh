#!/bin/bash

# Example usage: ./generate.sh ./from_top_doc

OUTPUT_DIR=$1

LOG_FILE=$OUTPUT_DIR/generate.log

rm -rf "$OUTPUT_DIR"
mkdir "$OUTPUT_DIR"

for N_SLOTS in 2 4 8 16 32 64 128 256 512
do
  echo -e "Using $N_SLOTS slots..." >> "$LOG_FILE"
  python init_topic_knowledge.py $N_SLOTS "$OUTPUT_DIR"/$N_SLOTS.pt >> "$LOG_FILE"
done
