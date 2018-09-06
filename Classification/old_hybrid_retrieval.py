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
import argparse 
import os, sys
import numpy as np 
import random

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

##################### configure ###############################################
parser = argparse.ArgumentParser()
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

############################### utile func ####################################
def format_query_candidates(query, candidates):
    """format the query and its candidates from ES,
        mapping text to ids, 
    """
    # words to ids, lack of tokenization, 
    len_q, id_query = line_to_id(query, vocab, max_len=15)
    pairs = []
    for candidate in candidates:
        len_c, id_c = line_to_id(candidate, vocab, max_len=15, pre_trunc=False)
        pairs.append([len_q, id_query, len_c, id_c])
    return pairs 
    
def generate_batch(pairs):
    """
    """
    t = list(zip(*pairs))
    outputs = None 
    for i in range(0, len(t)-1,2):
        txt_len, txt =(t[i], t[i+1])
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
    if use_gpu: # add flag 
        q_src, q_src_len, r_src, r_src_len = (q_src.cuda(), q_src_len.cuda(),
            r_src.cuda(), r_src_len.cuda())
    outputs, _ = model(q_src, q_src_len, r_src, r_src_len)
    print(outputs.size)
    return F.sigmoid(outputs) 
    
############################### Load model ####################################
if opt.model_from:
    print("Loading checkpoint from {}".format( opt.model_from))
    checkpoint = torch.load(opt.model_from,
        map_location=lambda storage, loc:storage)
    model_opt = checkpoint['opt']

print("building model ... ...")
model = ModelConstructor.make_base_model(model_opt, use_gpu(opt), checkpoint)


############################ build connection with ES ########################
es = connet_es()
print("connect to the chat server")
################################## Conversations ##############################
print("**************** start conversation [push Cntl-D to exit] *************")

while True:
    input_str = raw_input("U: ")
    # retrieve the ES 
    candidates = _do_query_use_file_info(es, input_str)
    # catch Exceptions
    if candidates is None or len(candidates) < 1:
        sys_output = (0., "Please input something else.")
        continue 
    else:
        sys_output = random.sample(candidates, 1)[0]
    print(">> \t{}\t{} :S1".format(sys_output[1], sys_output[0]))
    ############################# Re-Rank using a CNN-based Ranker#############
    candidate_replies = [c[1] for c in candidates] 
    pairs = format_query_candidates(input_str, candidate_replies)
    batch = generate_batch(pairs) 
    # score 
    scores = get_score(model, batch)
    scores = scores.data.cpu().numpy()
    # ranking candidates according to the prediction of the model.
    rank = np.argsort(scores)
    rank = rank[::-1].tolist()
    #
    index = random.sample(rank[:10], 1)[0] 
    print(">> \t{}\t{} :S2".format(candidate_replies[index], scores[index]))


    
