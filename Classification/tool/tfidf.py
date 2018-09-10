# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function

import io
import os
import sys
import logging
import numpy as np



logger = logging.getLogger(__name__)

'''
Build word's idf value for train corpus
and compute tfidf value
'''

class TFIDF:
    def __init__(self):
        self.idf_dict = {}
        # text_count, for computing idf
        self.text_count = 0

    def build_idf(self, stop_word_obj, opt):
        corpus_list = opt.train_corpus_path

        if opt.with_label:
            corpus_list = corpus_list[:-1]

        word_text_dict = {}

        for corpus_path in corpus_list:
            # check there exists the corpus_path or not.
            if not os.path.exists(corpus_path):
                logger.info('The path {} doese not exist for building vocabulary'.format(corpus_path))
                continue

            with io.open(corpus_path, "r", encoding='utf-8') as f:
                for line in f:
                    # words = line.strip().replace('\t', ' ').split()
                    words = stop_word_obj.remove_words(line.strip().replace('\t', ' '))
                    words_set = set(words)
                    for word in words_set:
                        word_text_dict.setdefault(word, 0)
                        word_text_dict[word] += 1

                    self.text_count += 1

        for word, word_text_count in word_text_dict.iteritems():
            self.idf_dict[word] = np.log(self.text_count * 1.0 / word_text_count)
            print('word: {}, idf: {}'.format(word, self.idf_dict[word]))

    # def compute_tfidf(self, word, tf):
    #     tfidf_value = tf * np.log(self.text_count / self.idf_dict[word])
    #     return tfidf_value


    def get_idf(self, word, default=0.0):
        '''
        Obtain word's idf value, in order to compute word's tfidf value.
        :param word:
        :param default:
        :return:
        '''
        self.idf_dict.get(word, default)
