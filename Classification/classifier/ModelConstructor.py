"""
This file is for models creation, which consults options
and creates each encoder and decoder accordingly.
"""
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn

from modules.Encoder import MeanEncoder, RNNEncoder, CNNEncoder
from Models import SingleArch, DoubleArch, ExtendDoubleArch
from Models import PairwiseRanker
from torch.nn.init import xavier_uniform

from modules.Embeddings import Embeddings
from modules.utils import use_gpu

def make_embeddings(opt, num_word, padding_idx):
    """
    Make an Embeddings instance.
    Args:
        opt: the option in current environment.
        num_word, 
        padding_idx,
        for_encoder(bool): make Embeddings for encoder or decoder?
    """
    embedding_dim = opt.word_vec_size
    word_padding_idx = padding_idx
    num_word_embeddings = num_word

    return Embeddings(word_vec_size=embedding_dim,
                      dropout=opt.dropout,
                      word_padding_idx=word_padding_idx,
                      word_vocab_size=num_word_embeddings,
                      sparse=opt.optim == "sparseadam")

def make_encoder(opt, embeddings):
    """
    Various encoder dispatcher function.
    Args:
        opt: the option in current environment.
        embeddings (Embeddings): vocab embeddings for this encoder.
    """
    # "rnn" or "brnn" 
    if opt.encoder_type == 'rnn':
        return RNNEncoder(opt.rnn_type, opt.brnn, opt.enc_layers,
                      opt.rnn_size, opt.dropout, embeddings=embeddings)
    elif opt.encoder_type == 'cnn':
        return CNNEncoder(opt.hidden_size, num_layers=opt.enc_layers, 
                    filter_num=opt.filter_num, filter_sizes=[1, 2, 3, 4], 
                    dropout=opt.dropout, embeddings=embeddings)
    elif opt.encoder_type == "mean":
        return MeanEncoder(opt.enc_layers, embeddings)
    else:
        raise ValueError("Unsupported Encoder type: {0}".format(opt.encoder_type))

def make_base_model(model_opt, gpu, checkpoint=None):
    """
    Args:
        model_opt: the option loaded from checkpoint.
        gpu(bool): whether to use gpu.
        checkpoint: the model gnerated by train phase, or a resumed snapshot
                    model from a stopped training.
    Returns:
        the Classifier.
    """

    # Make encoder.
    embeddings = make_embeddings(model_opt, 
                                 model_opt.numwords,
                                 model_opt.padding_idx )
    encoder = make_encoder(model_opt, embeddings)

    # Make classification model
    if model_opt.class_num == 2:
        output_size = 1
    else:
        output_size = model_opt.class_num

    # get the input size of the classifier
    if model_opt.encoder_type == "cnn":
        input_size = model_opt.hidden_size #encoder.pool_feat_num
    else:
        input_size = model_opt.rnn_size
    
    # classifier architecture, 'single', 'double', 'exdouble'
    if model_opt.cls_arch == 'single': 
        model = SingleArch(encoder, input_size, model_opt.hidden_size,
           output_size, dropout=0.0, score_fn_type=model_opt.score_fn_type)
    elif model_opt.cls_arch == 'double':
        model = DoubleArch(encoder, input_size, model_opt.hidden_size,
           output_size, dropout=0.0, score_fn_type=model_opt.score_fn_type)
    elif model_opt.cls_arch == 'exdouble':
        model = ExtendDoubleArch(encoder, input_size, model_opt.hidden_size,
           output_size, dropout=0.0, score_fn_type=model_opt.score_fn_type,
           bilinear_flag=model_opt.bilinear_flag, dot_flag=model_opt.dot_flag,
           substract_flag=model_opt.substract_flag, 
           inner_prod_flag=model_opt.inner_prod_flag)
    elif model_opt.cls_arch == 'pairwise':
        model = PairwiseRanker(encoder, input_size, model_opt.hidden_size, 
                       output_size, score_fn_type=model_opt.score_fn_type)
    else:
        raise ValueError("""The Classification Architecture
                         {} is invalid""".format(model.cls_arch)) 
    model.model_type = model_opt.model_type

    # Load the model states from checkpoint or initialize them.
    if checkpoint is not None:
        print('Loading model parameters.')
        model.load_state_dict(checkpoint['model'])
    else:
        if model_opt.param_init != 0.0:
            print('Intializing model parameters.')
            for p in model.parameters():
                p.data.uniform_(-model_opt.param_init, model_opt.param_init)
        for p in model.parameters():
            if p.dim() > 1:
                xavier_uniform(p)
    # Make the whole model leverage GPU if indicated to do so.
    if gpu:
        model.cuda()
    else:
        model.cpu()

    return model
