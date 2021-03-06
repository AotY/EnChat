#!/usr/bin/env bash
#python preprocess.py -train_corpus_path /home/xuzhen/Project/dataset/opensub/open_train_post.txt /home/xuzhen/Project/dataset/opensub/open_train_response.txt -valid_corpus_path /home/xuzhen/Project/dataset/opensub/open_valid_post.txt /home/xuzhen/Project/dataset/opensub/open_valid_response.txt -save_data data/opensub/opensub -max_shard_size 15360000 -vocab_path data/opensub/vocab.pt -seq_length 15 -vocab_size 50000 -pre_word_vecs_path /home/xuzhen/Project/dataset/w2v/glove_twitter/glove.twitter.27B.100d.txt
python preprocess.py
    -train_corpus_path /home/xuzhen/Project/dataset/douban/gan/train_post.txt /home/xuzhen/Project/dataset/douban/gan/train_response.txt \
    -valid_corpus_path /home/xuzhen/Project/dataset/douban/gan/valid_post.txt /home/xuzhen/Project/dataset/douban/gan/valid_response.txt \
    -save_data data/douban/douban \
    -max_shard_size 15360000 \
    -vocab_path data/douban/vocab.pt \
    -seq_length 15 -vocab_size 50000 \
    -pre_word_vecs_path /home/xuzhen/Project/dataset/douban/w2v/chat.w2v.35w.100d.clean.txt \

/