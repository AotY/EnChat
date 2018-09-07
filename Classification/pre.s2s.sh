#!/usr/bin/env bash
python preprocess.py \
    -train_corpus_path /home/Research/EnChat/askqa/train.post.txt /home/Research/EnChat/askqa/train.response.txt /home/Research/EnChat/askqa/train.label.txt \
    -valid_corpus_path /home/Research/EnChat/askqa/valid.post.txt /home/Research/EnChat/askqa/valid.response.txt /home/Research/EnChat/askqa/valid.label.txt \
    -valid_corpus_path /home/Research/EnChat/askqa/test.post.txt /home/Research/EnChat/askqa/test.response.txt /home/Research/EnChat/askqa/test.label.txt \
    -save_data data/reddit/askqa \
    -max_shard_size 153600000 \
    -vocab_path data/reddit/askqa/vocab.pt \
    -seq_length 80 \
    -vocab_size 100000 \
    -with_label \
    -pre_word_vecs_path /home/taoqing/Research/data/GoogleNews-vectors-negative300.bin \
    -pre_word_vecs_type word2vec \
    -pre_word_vecs_dim 300 \
    -pre_word_vecs_file_type binary \
    -pre_trained_vocab_embedding_file data/reddit/askqa/vocab_word2vec.300d.npy \

/

