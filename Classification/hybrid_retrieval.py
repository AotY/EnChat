#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ES + CNN classifier for retrieval-based chat-bot 
    Input, a query
    1) Retrieving the input query in ES, and then several similar queries will be
    recalled. Responses of the recalled queries can be taken as the candidate replies 
    of the input query. 
    2) After that, a CNN-based classifier is adopted to rank candidates, as to 
    select a better response for the input query. 

    Here this script is to unite the two strategies. 
"""
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import numpy as np
import random
import logging
import datetime

import torch
import torch.nn.functional as F
from torch import cuda
from torch.autograd import Variable

import classifier.opts as opts
from classifier import ModelConstructor
from modules.utils import use_gpu
from tool.func_utils import line_to_id
from tool.func_utils import sentence_pad

from search_es import connet_es
from search_es import _do_query_use_file_info
from embedding.embedding_score import get_avg_embedding_score, get_tfidf_embedding_score, get_wmd_embedding_score, \
    get_extreme_embedding_score, get_greedy_embedding_score
from tool.tfidf import TFIDF
from tool.remove_stop_words import StopWord
from gensim.models import KeyedVectors

##################### running logger #########################################
runlog_name = "log/chatting.{}.log".format(datetime.date.today())
run_logger = logging.getLogger('runlog')
run_logger.setLevel(logging.INFO)

runHandler = logging.FileHandler(runlog_name, 'a')
runHandler.setLevel(logging.INFO)

run_logger.addHandler(runHandler)

##################### configure ###############################################
# get optional parameters

# parser = argparse.ArgumentParser()
parser = argparse.ArgumentParser(description='hybrid_retrieval.py',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

opts.model_opts(parser)
opts.train_opts(parser)
opts.test_opts(parser)

opt = parser.parse_args()
opt.brnn = (opt.encoder_type == "brnn")

# set gpu 
if opt.gpuid:
    cuda.set_device(opt.gpuid[0])
    if opt.seed > 0:
        torch.cuda.manual_seed(opt.seed)

# Load vocab and confirm opt.vocab_size
vocab = torch.load(opt.vocab_path)
opt.padding_idx = 0
opt.numwords = len(vocab.word2idx)

# Load pre-trained embedding for vocab
# pre_trained_embedding = np.load(opt.pre_trained_vocab_embedding)

# Load gensim model
print('Loading Gensim model.')
if opt.binary:
    gensim_model = KeyedVectors.load_word2vec_format(opt.pre_trained_vocab_embedding, binary=True)
else:
    gensim_model = KeyedVectors.load_word2vec_format(opt.pre_trained_vocab_embedding, binary=False)

print ('vocab_size: {}'.format(len(gensim_model.vocab)))
# print ('word: clouds, embedding: {}'.format(gensim_model.wv['clouds']))

# Load TFIDF object, for computing word's tfidf value.
print('Loading tfidf object.')
tfidf = torch.load(opt.vocab_tfidf)

# Load stop_words
print('Init stop words object.')
stop_word_obj = StopWord(opt.stop_word_file)


############################### utile func ####################################
def format_query_candidates(query, candidates):
    """format the query and its candidates from ES,
        mapping text to ids, 
    """
    # words to ids, lack of tokenization, 
    len_query, id_query = line_to_id(query, vocab, max_len=15, lower=opt.lower)

    pairs = []
    for candidate in candidates:
        len_candidate, id_candidate = line_to_id(candidate, vocab, max_len=15, pre_trunc=False, lower=opt.lower )
        pairs.append([len_query, id_query, len_candidate, id_candidate])
    return pairs


# pairs: [[len_query, id_query, len_candidate, id_candidate], [len_query, id_query, len_candidate, id_candidate]]
def generate_batch(pairs):
    """
    """
    t = list(zip(*pairs))
    outputs = None

    for i in range(0, len(t) - 1, 2):
        txt_len, txt = (t[i], t[i + 1])

        max_len = max(txt_len)

        txt = sentence_pad(txt, max_len=max_len)

        txt = Variable(torch.from_numpy(txt.T.astype(np.int64)))

        txt_len = torch.LongTensor(txt_len).view(-1)

        if outputs is None:
            outputs = (txt, txt_len)
        else:
            outputs = outputs + (txt, txt_len)

    return outputs


def get_score(model, batch):
    """
    """
    model.eval()
    q_src, q_src_len, r_src, r_src_len = batch
    if use_gpu:  # add flag
        q_src, q_src_len, r_src, r_src_len = (q_src.cuda(), q_src_len.cuda(),
                                              r_src.cuda(), r_src_len.cuda())

    outputs, _ = model(q_src, q_src_len, r_src, r_src_len)

    print(outputs.size)

    try:
        return torch.sigmoid(outputs)
    except AttributeError:
        return F.sigmoid(outputs)


############################### Load model ####################################
if opt.model_from:
    print("Loading checkpoint from {}".format(opt.model_from))
    checkpoint = torch.load(opt.model_from,
                            map_location=lambda storage, loc: storage)
    model_opt = checkpoint['opt']

print("building model ... ...")
model = ModelConstructor.make_base_model(model_opt, use_gpu(opt), checkpoint)

############################ build connection with ES ########################
# es = connet_es('10.0.1.12', '9200')
es = connet_es('127.0.0.1', '9200')

print("connect to the chat server")
################################## Conversations ##############################
print("**************** start conversation [push Cntl-D to exit] *************")

while True:
    input_str = raw_input("U: ")

    # retrieve the ES
    candidates = _do_query_use_file_info(es, input_str)

    # logger
    run_logger.info("query: " + input_str)
    for idx, c in enumerate(candidates[:10]):
        run_logger.info("ES, c{}, {}, {}".format(idx, c[1], c[0]))

    # catch Exceptions
    if candidates is None or len(candidates) < 1:
        sys_output = (0., "Please input something else.")
        continue
    else:
        sys_output = random.sample(candidates[:10], 1)[0]
    print(">> \t{}\t{} :S1".format(sys_output[1], sys_output[0]))

    ############################# Re-Rank using a CNN-based Ranker#############
    candidate_replies = [c[1] for c in candidates]
    pairs = format_query_candidates(input_str, candidate_replies)

    batch = generate_batch(pairs)

    # score
    cnn_scores = get_score(model, batch)
    cnn_scores = cnn_scores.data.cpu().numpy()

    # ranking candidates according to the prediction of the model.
    cnn_rank = np.argsort(cnn_scores)
    cnn_rank = cnn_rank[::-1].tolist()

    for idx, c_idx in enumerate(cnn_rank[:10]):
        run_logger.info("CNN Ranker, c{}, {}, {}".format(idx, candidate_replies[c_idx], cnn_scores[c_idx]))

    index = random.sample(cnn_rank[:10], 1)[0]
    print(">> \t{}\t{} :S_cnn".format(candidate_replies[index], cnn_scores[index]))

    ############################# Re-Rank using a Word Embedding -based Ranker#############
    # pairs [[len_q, id_query, len_c, id_c]]
    # batch []
    # vector_query = pairs[0][1]
    # matrix_candidate = [pair[-1] for pair in pairs]

    avg_embedding_score = get_avg_embedding_score(vocab, gensim_model, input_str, candidate_replies, stop_word_obj, opt.lower)  # opt.embedding_ranker_type
    avg_embedding_rank = np.argsort(avg_embedding_score)
    avg_embedding_rank = avg_embedding_rank[::-1].tolist()

    for idx, e_idx in enumerate(avg_embedding_rank[:10]):
        run_logger.info(
            "Avg Embedding Ranker, c{}, {}, {}".format(idx, candidate_replies[e_idx], avg_embedding_score[e_idx]))
    index = random.sample(avg_embedding_rank[:10], 1)[0]
    print(">> \t{}\t{} :S_avg".format(candidate_replies[index], avg_embedding_score[index]))

    ############################# Re-Rank using a Word Embedding-based with TFIDF weight Ranker#############
    tfidf_embedding_score = get_tfidf_embedding_score(vocab, gensim_model, tfidf, input_str, candidate_replies, stop_word_obj, opt.lower)
    tfidf_embedding_rank = np.argsort(tfidf_embedding_score)
    tfidf_embedding_rank = tfidf_embedding_rank[::-1].tolist()
    for idx, te_idx in enumerate(tfidf_embedding_rank[:10]):
        run_logger.info(
            "TFIDF Embedding Ranker, c{}, {}, {}".format(idx, candidate_replies[te_idx], tfidf_embedding_score[te_idx]))
    index = random.sample(tfidf_embedding_rank[:10], 1)[0]
    print(">> \t{}\t{} :S_tfidf".format(candidate_replies[index], tfidf_embedding_score[index]))


    ############################# Re-Rank using a Word Embedding-based with WMD (EMD) Ranker#############
    wmd_embedding_score = get_wmd_embedding_score(gensim_model, input_str, candidate_replies, stop_word_obj, opt.lower)
    wmd_embedding_rank = np.argsort(wmd_embedding_score)
    wmd_embedding_rank = wmd_embedding_rank[::-1].tolist()

    for idx, w_idx in enumerate(wmd_embedding_rank[:10]):
        run_logger.info(
            "WMD Embedding Ranker, c{}, {}, {}".format(idx, candidate_replies[w_idx], wmd_embedding_score[w_idx]))
    index = random.sample(wmd_embedding_rank[:10], 1)[0]
    print(">> \t{}\t{} :S_wmd".format(candidate_replies[index], wmd_embedding_score[index]))

    ############################# Re-Rank using a Word Embedding-based with Extreme Ranker#############
    extreme_embedding_score = get_extreme_embedding_score(vocab, gensim_model, input_str, candidate_replies, stop_word_obj, opt.lower)
    extreme_embedding_rank = np.argsort(extreme_embedding_score)
    extreme_embedding_rank = extreme_embedding_rank[::-1].tolist()

    for idx, e_idx in enumerate(extreme_embedding_rank[:10]):
        run_logger.info(
            "Extreme Embedding Ranker, c{}, {}, {}".format(idx, candidate_replies[e_idx], extreme_embedding_score[e_idx]))
    index = random.sample(extreme_embedding_rank[:10], 1)[0]
    print(">> \t{}\t{} :S_extreme".format(candidate_replies[index], extreme_embedding_score[index]))


    ############################# Re-Rank using a Word Embedding-based with Extreme Ranker#############
    greedy_embedding_score = get_greedy_embedding_score(vocab, gensim_model, input_str, candidate_replies, stop_word_obj, opt.lower)
    greedy_embedding_rank = np.argsort(greedy_embedding_score)
    greedy_embedding_rank = greedy_embedding_rank[::-1].tolist()

    for idx, e_idx in enumerate(greedy_embedding_rank[:10]):
        run_logger.info(
            "Extreme Embedding Ranker, c{}, {}, {}".format(idx, candidate_replies[e_idx], greedy_embedding_score[e_idx]))
    index = random.sample(greedy_embedding_rank[:10], 1)[0]
    print(">> \t{}\t{} :S_greedy".format(candidate_replies[index], greedy_embedding_score[index]))


    ############################# Hybird #############################
    # print('cnn_scores: {} '.format(cnn_scores))
    # print('avg_embedding_score: {} '.format(avg_embedding_score))
    # print('tfidf_embedding_score: {} '.format(tfidf_embedding_score))
    # print('WMD embedding_score: {} '.format(wmd_embedding_score))
    hybrid_score = np.mean([cnn_scores,
                            avg_embedding_score,
                            extreme_embedding_score,
                            greedy_embedding_score,
                            tfidf_embedding_score,
                            wmd_embedding_score
                            ],
                           axis=0)
    # print('hybrid_score: {} '.format(hybrid_score))
    hybrid_embedding_rank = np.argsort(hybrid_score)
    hybrid_embedding_rank = hybrid_embedding_rank[::-1].tolist()
    for idx, h_idx in enumerate(hybrid_embedding_rank[:10]):
        run_logger.info(
            "Hybrid Ranker, c{}, {}, {}".format(idx, candidate_replies[h_idx], hybrid_score[h_idx]))
    index = random.sample(hybrid_embedding_rank[:10], 1)[0]
    print(">> \t{}\t{} :S_hybrid".format(candidate_replies[index], hybrid_score[index]))
