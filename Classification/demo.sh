#!/usr/bin/env bash
python hybrid_retrieval.py \
    -word_vec_size 100 \
    -encoder_type cnn \
    -class_num 2 \
    -data data/reddit \
    -save_model model/askqa \
    -batch_size 256 \
    -optim adam \
    -learning_rate 0.0001 \
    -valid_batch_size 128 \
    -cls_arch exdouble \
    -score_fn_type MLP \
    -dot_flag \
    -inner_prod_flag \
    -padding_idx 0 \
    -enc_layers 1 \
    -hidden_size 256 \
    -report_every 500 \
    -gpuid 5 \
    -seed 7 \
    -vocab_path data/reddit/vocab_lower.pt \
    -model_from model/askqa/askqa_acc_84.07_ppl_0.35_e3.pt \
    -save_data data/reddit \
    -vocab_tfidf data/reddit/vocab.tfidf.pt \
    -pre_trained_vocab_embedding data/reddit/vocab_word2vec.300d.txt \
    -binary False \
    -lower \


/

