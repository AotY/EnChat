# -*- coding: utf-8 -*-

'''
train word embedding using gensim

data from reddit
'''
from __future__ import division
from __future__ import print_function

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

import gensim
from gensim.models import Word2Vec

# define training data
sentences = [['this', 'is', 'the', 'first', 'sentence', 'for', 'word2vec'],
             ['this', 'is', 'the', 'second', 'sentence'],
             ['yet', 'another', 'sentence'],
             ['one', 'more', 'sentence'],
             ['and', 'the', 'final', 'sentence']]

documents = sentences
# build vocabulary and train model
model = gensim.models.Word2Vec(
    documents,
    size=300,
    window=10,
    min_count=5,
    sample=1e-3,
    negative=5,
    cbow_mean=1,
    hs=0,
    workers=10
)

# train model
model.train(documents, total_examples=len(documents), epochs=10)

# fname, fvocab=None, binary=False
model.save_word2vec_format('reddit.300d', fvocab=None, binary=True)

# summarize the loaded model
print(model)
# summarize vocabulary

# access vector for one word
print(model['reddit'])
# save model
# model.save('reddit.gensim.bin')

# load model
# new_model = Word2Vec.load('model.bin')
# print(new_model)
