#!/usr/bin/env python

from __future__ import division
from __future__ import print_function


import argparse
import glob
import os
import sys
import random
from datetime import datetime

import torch
import torch.nn as nn
from torch import cuda
# import torch.optim as optim

import classifier.Models
from classifier import ModelConstructor
import classifier.opts as opts
from tool.vocab import Vocab 
from modules.utils import use_gpu
from classifier.Trainer import Statistics, SingleTrainer, DoubleTrainer
from modules.Optim import Optim
from tool.datasetbase import DataSet, TestDataset
import numpy as np

parser = argparse.ArgumentParser(
    description='train.py',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

opts.model_opts(parser)
opts.train_opts(parser)
opts.test_opts(parser)

opt = parser.parse_args()

opt.brnn = (opt.encoder_type == "brnn")
if opt.seed > 0:
    random.seed(opt.seed)
    torch.manual_seed(opt.seed)

if torch.cuda.is_available() and not opt.gpuid:
    print("WARNING: You have a CUDA device, should run with -gpuid 0")

if opt.gpuid:
    cuda.set_device(opt.gpuid[0])
    if opt.seed > 0:
        torch.cuda.manual_seed(opt.seed)

if opt.tensorboard:
    from tensorboardX import SummaryWriter
    writer = SummaryWriter(
        opt.tensorboard_log_dir + datetime.now().strftime("/%b-%d_%H-%M-%S"),
        comment="Onmt")

# modify vocab size according to the real vocab
vocab = torch.load(opt.vocab_path)
# 
opt.padding_idx = 0
opt.numwords = len(vocab.word2idx)

# loss function
criterion = nn.BCELoss(size_average=False) # nn.NLLLoss(weight, size_average=False)

progress_step = 0

def report_func(epoch, batch, num_batches,
                progress_step,
                start_time, lr, report_stats):
    """
    This is the user-defined batch-level traing progress
    report function.

    Args:
        epoch(int): current epoch count.
        batch(int): current batch count.
        num_batches(int): total number of batches.
        progress_step(int): the progress step.
        start_time(float): last report time.
        lr(float): current learning rate.
        report_stats(Statistics): old Statistics instance.
    Returns:
        report_stats(Statistics): updated Statistics instance.
    """
    if batch % opt.report_every == -1 % opt.report_every:
        report_stats.output(epoch, batch + 1, num_batches, start_time)
        report_stats = Statistics()
    return report_stats

############################*****************************#########################

def test_model(model, optim, criterion, model_opt):
    trainer = DoubleTrainer(model, optim, criterion, True)

    print('\nStart training...')
    print(' * number of epochs: %d, starting from Epoch %d' %
          (opt.epochs + 1 - opt.start_epoch, opt.start_epoch))
    print(' * batch size: %d' % opt.batch_size)
    
    # testing 
    # 2. Validate on the validation set.
    # valid_iter = lazily_load_dataset("valid")
    vocab = torch.load(opt.vocab_path)
    print(vocab.word2idx.keys()[:5])
    valid_iter = TestDataset(vocab, 
                             opt.test_corpus_path,
                             opt.batch_size,
                             max_len=50, pre_trunc=True)
    valid_stats = trainer.test(valid_iter)
    print('Validation Loss: %g' % valid_stats.xent())
    print('Validation accuracy: %g' % valid_stats.accuracy())
    if len(valid_stats.scores):
        print('Validation recall_ks: {}'.format( valid_stats.recall_ks()) )
    # write the test result into file  
    # load test_opt 
    with open(opt.test_output, 'w') as fout:
        for score in valid_stats.scores:
            fout.write('{:.4f}\n'.format(score))
    # end 

def lazily_load_dataset(corpus_type):
    """
    Dataset generator. Don't do extra stuff here, like printing,
    because they will be postponed to the first loading time.

    Args:
        corpus_type: 'train' or 'valid'
    Returns:
        A list of dataset, the dataset(s) are lazily loaded.
    """
    assert corpus_type in ["train", "valid"]

    def lazy_dataset_loader(pt_file, corpus_type):
        dataset = torch.load(pt_file)
        print('Loading %s dataset from %s, number of examples: %d' %
              (corpus_type, pt_file, len(dataset.examples)))
        return dataset

    # Sort the glob output by file name (by increasing indexes).
    pts = sorted(glob.glob(opt.data + '.' + corpus_type + '.[0-9]*.pt'))
    if pts:
        for pt in pts:
            dataset = lazy_dataset_loader(pt, corpus_type)
            dataset.set_property(batch_size=opt.batch_size, 
                                 with_label=True, rank=False)
            yield dataset
    else:
        # Only one onmt.io.*Dataset, simple!
        pt = opt.data + '.' + corpus_type + '.pt'
        dataset = lazy_dataset_loader(pt, corpus_type)
        dataset.set_batch_size(opt.batch_size)
        yield dataset

######################********************#########################

def check_save_model_path():
    save_model_path = os.path.abspath(opt.save_model)
    model_dirname = os.path.dirname(save_model_path)
    if not os.path.exists(model_dirname):
        os.makedirs(model_dirname)

def tally_parameters(model):
    n_params = sum([p.nelement() for p in model.parameters()])
    print('* number of parameters: %d' % n_params)
    enc = 0
    dec = 0
    for name, param in model.named_parameters():
        if 'encoder' in name:
            enc += param.nelement()
        else:
            dec += param.nelement()
    print('encoder: ', enc)
    print('project: ', dec)

def build_optim(model, opt, checkpoint):
    if opt.train_from:
        print('Loading optimizer from checkpoint.')
        optim = checkpoint['optim']
        optim.optimizer.load_state_dict(
            checkpoint['optim'].optimizer.state_dict())
    else:
        print('Making optimizer for training.')
        optim = Optim(
            opt.optim, opt.learning_rate, opt.max_grad_norm,
            lr_decay=opt.learning_rate_decay,
            start_decay_at=opt.start_decay_at,
            beta1=opt.adam_beta1,
            beta2=opt.adam_beta2,
            adagrad_accum=opt.adagrad_accumulator_init,
            decay_method=opt.decay_method,
            warmup_steps=opt.warmup_steps,
            model_size=opt.rnn_size)

    optim.set_parameters(model.parameters())

    return optim

def build_model(model_opt, opt, checkpoint):
    print('Building model...')
    model = ModelConstructor.make_base_model(model_opt,
                                             use_gpu(opt),
                                             checkpoint)
    print(model)
    return model

def main():
    # Load checkpoint if we resume from a previous training.
    if opt.train_from:
        print('Loading checkpoint from %s' % opt.train_from)
        checkpoint = torch.load(opt.train_from,
                                map_location=lambda storage, loc: storage)
        model_opt = checkpoint['opt']
        # I don't like reassigning attributes of opt: it's not clear.
        opt.start_epoch = checkpoint['epoch'] + 1
    else:
        checkpoint = None
        model_opt = opt

    # Build model.
    model = build_model(model_opt, opt, checkpoint)
    tally_parameters(model)
    check_save_model_path()
    # 
    # if opt.pre_word_vecs is not None and os.path.exists(opt.pre_word_vecs):
    #     W = np.load(opt.pre_word_vecs)
    #     print('Loading embedding from {}'.format(opt.pre_word_vecs))
    #     model.encoder.embeddings.embeddings.weight.data.copy_(torch.from_numpy(W))

    # Build optimizer.
    optimizer = build_optim(model, opt, checkpoint)

    # Do training.
    test_model(model, optimizer, criterion, model_opt)

if __name__ == "__main__":
    main()
