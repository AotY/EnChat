# -*- coding: utf-8 -*-

'''
load pre-trained word embedding
google word2vec or
standford glove
'''
from __future__ import division
from __future__ import print_function

import torch
import numpy as np
from pybloom import BloomFilter

'''
load word2vec
binary
'''


def load_word2vec(vocab, vocab_size, vec_file, embedding_dim, pre_word_vecs_file_type):
    # init
    pre_trained_embedding = np.random.uniform(-0.25, 0.25, (vocab_size, embedding_dim))

    # load any vectors from the word2vec
    print("Load word2vec file {}\n".format(vec_file))

    if pre_word_vecs_file_type == 'binary':
        model = 'rb'
    else:
        model = 'r'

    in_bf = BloomFilter(capacity=3000000, error_rate=0.001)

    with open(vec_file, model) as f:
        header = f.readline()

        word2vec_vocab_size, word2vec_embedding_dim = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * word2vec_embedding_dim

        for line in xrange(word2vec_vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break

                if ch != '\n':
                    word.append(ch)

            # word2idx
            idx = vocab.word2idx.get(word, vocab.unkid)

            if idx != vocab.unkid:
                in_bf.add(word)
                pre_trained_embedding[idx] = np.fromstring(f.read(binary_len), dtype='float32')
            else:
                f.read(binary_len)

    return in_bf.count, pre_trained_embedding


'''
load glove
binary
'''


def load_glove(vocab, vocab_size, glove_file, glove_vocab_size, embedding_dim, pre_word_vecs_file_type):
    # init
    pre_trained_embedding = np.random.uniform(-0.25, 0.25, (vocab_size, embedding_dim))

    # load any vectors from the word2vec
    print("Load glove file {}\n".format(glove_file))

    in_bf = BloomFilter(capacity=3000000, error_rate=0.001)

    if pre_word_vecs_file_type == 'binary':
        model = 'rb'
    else:
        model = 'r'
    with open(glove_file, model) as f:
        binary_len = np.dtype('float32').itemsize * embedding_dim

        for line in xrange(glove_vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break

                if ch != '\n':
                    word.append(ch)
            # word2idx
            idx = vocab.word2idx.get(word, vocab.unkid)

            if idx != vocab.unkid:
                in_bf.add(word)
                pre_trained_embedding[idx] = np.fromstring(f.read(binary_len), dtype='float32')
            else:
                f.read(binary_len)

    return in_bf.count, pre_trained_embedding


def save_reddit_embedding(embedding, save_file):
    # obj, f, pickle_module=pickle, pickle_protocol=DEFAULT_PROTOCOL
    torch.save(embedding, save_file)


if __name__ == '__main__':
    # Load vocab and confirm opt.vocab_size
    vocab_path = './../data/reddit/vocab.pt'
    vocab = torch.load(vocab_path)
    padding_idx = 0
    vocab_size = len(vocab.word2idx)

    embedding_dim = 300
    initW = load_word2vec(vocab, vocab_size, vocab_path, embedding_dim)

    save_file = './../data/reddit/reddit_embedding_word2vec.pt'

    save_reddit_embedding(initW, save_file)

    pass



