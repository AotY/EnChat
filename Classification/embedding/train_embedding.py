# -*- coding: utf-8 -*-

'''
train word embedding using gensim

data from reddit
'''
from __future__ import division
from __future__ import print_function

import io
import argparse
import logging
import multiprocessing
import os
import sys
from gensim.models.word2vec import LineSentence
from gensim.models.word2vec import Word2Vec
from train_embedding_opt import train_embedding_opt


def load_source(corpus_path_list, logger):
    source = []
    for corpus_path in corpus_path_list:
        # check there exists the corpus_path or not.
        if not os.path.exists(corpus_path):
            logger.info('The path {} doese not exist for building vocabulary'.format(corpus_path))
            continue

        logger.info("corpus_path: %s .", corpus_path)

        with io.open(corpus_path, "r", encoding='utf-8') as f:
            for line in f:
                source.append(line.strip().replace('\t', ' '))

    return source


# a memory-friendly iterator
class MySentences(object):
    def __init__(self, corpus_path_list, max_words):
        self.corpus_path_list = corpus_path_list
        self.max_words = max_words

    def __iter__(self):
        for corpus_path in self.corpus_path_list:
            # check there exists the corpus_path or not.

            logger.info("corpus_path: %s .", corpus_path)

            with io.open(corpus_path, "r", encoding='utf-8') as f:
                for line in f:
                    words = line.strip().replace('\t', ' ').split()
                    yield words[:self.max_words]


if __name__ == '__main__':

    program = os.path.basename(sys.argv[0])

    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("Running %s", ' '.join(sys.argv))

    # get optional parameters
    parser = argparse.ArgumentParser(description='train_embedding.py',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    train_embedding_opt(parser)
    opt = parser.parse_args()

    # Check and process input arguments.
    max_words = opt.max_words

    logger.info("Max article length: %s words.", max_words)

    params = {
        'size': opt.size,
        'window': opt.window,
        'min_count': opt.min_count,
        'workers': max(1, multiprocessing.cpu_count() - 2),
        'sample': opt.sample,
        'alpha': opt.alpha,
        'hs': opt.hs,
        'negative': opt.negative,
        'iter': opt.epochs
    }

    source = load_source(opt.corpus_path_list, logger)

    # word2vec = Word2Vec(LineSentence(source, max_sentence_length=max_length),
    #                     **params)

    word2vec = Word2Vec(MySentences(opt.corpus_path_list, max_words=max_words),
                        **params)

    vocab_size = len(word2vec.wv.vocab)
    vocab_size_str = str(vocab_size)

    word2vec.save(opt.save_path + '/reddit_word2vec.model') # Save the model.

    # save word2vec
    word2vec.wv.save_word2vec_format(
        opt.save_path + '/reddit.w2v.{}.{}.{}d.txt'.format(vocab_size_str[0], len(vocab_size_str), opt.size),
        binary=opt.binary)

    pass