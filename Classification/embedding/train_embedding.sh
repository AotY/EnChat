#!/usr/bin/env bash
python train_embedding.py \
    -train_corpus_path /home/Research/EnChat/askqa/train.post.txt /home/Research/EnChat/askqa/train.response.txt \
    -save_path data/reddit/askqa \
    -binary False \
    -max_words 50 \
    -size 200 \
    -window 7 \
    -alpha 0.025 \
    -min_count 5 \
    -negative 7 \

/

