# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
from collections import Counter

'''
Get embedding score with tfidf weight
'''
def get_tfidf_embedding_score(vocab, pre_trained_embedding, vector_query, input_str, matrix_candidate, candidate_replies,  tfidf, stop_word_obj):

    # embedding dim
    embedding_dim = pre_trained_embedding.shape[1]

    # init
    query_tfidf_weights = np.array([round(1.0 / embedding_dim, 5)] * embedding_dim)

    query_words = stop_word_obj.remove_words(input_str.strip().replace('\t', ' '))
    query_len = len(query_words)
    # freq
    query_words_counter = Counter(query_words)

    # conpute tfidf
    query_words_set = set(query_words)
    for word in query_words_set:
        word_id = vocab.word2idx[word]
        if word == vocab.unkid:
            continue

        word_tfidf = (query_words_counter[word] / query_len) * tfidf.get(word, 0)

        if word_tfidf != 0:
            for idx, query_id in enumerate(vector_query):
                if query_id == word_id:
                    query_tfidf_weights[idx] =  word_tfidf
    '''
    for idx, word_id in enumerate(vector_query):
        word = vocab.idx2word[word_id]
        if word in query_words:
            word_tfidf = ( query_words_counter[word] / query_len) * tfidf.get(word, 0)
            if word_tfidf != 0:
                query_tfidf_weights[idx] = word_tfidf
    '''

    # scalar
    query_min_weight = np.min(query_tfidf_weights)
    query_max_weight = np.max(query_tfidf_weights)
    query_tfidf_weights = (query_tfidf_weights - query_min_weight) / (query_max_weight - query_max_weight)

    query_tfidf_weight_vector = np.zeros_like(query_tfidf_weights)

    for id_query, weight in zip(vector_query, query_tfidf_weights):
        query_tfidf_weight_vector += pre_trained_embedding[id_query] * weight

    candidate_tfidf_weight_matrix = []
    for candidate_replie, vector_candidate in zip(candidate_replies, matrix_candidate):
        candidate_tfidf_weights = np.array([round(1.0 / embedding_dim, 5)] * embedding_dim)
        candidate_words = stop_word_obj.remove_words(input_str.strip().replace('\t', ' '))
        candidate_len = len(query_words)
        candidate_words_counter = Counter(query_words)

        # conpute tfidf
        candidate_words_set = set(candidate_words)
        for word in candidate_words_set:
            word_id = vocab.word2idx[word]
            if word == vocab.unkid:
                continue

            word_tfidf = (candidate_words_counter[word] / candidate_len) * tfidf.get(word, 0)

            if word_tfidf != 0:
                for idx, can_id in enumerate(vector_candidate):
                    if can_id == word_id:
                        candidate_tfidf_weights[idx] = word_tfidf

        # scalar
        candidate_min_weight = np.min(candidate_tfidf_weights)
        candidate_max_weight = np.max(candidate_tfidf_weights)
        candidate_tfidf_weights = (candidate_tfidf_weights - candidate_min_weight) / (candidate_max_weight - candidate_min_weight)

        candidate_tfidf_weight_vector = np.zeros_like(candidate_tfidf_weights)

        for id_can, weight in zip(vector_query, query_tfidf_weights):
            candidate_tfidf_weight_vector += pre_trained_embedding[id_can] * weight

        candidate_tfidf_weight_matrix.append(candidate_tfidf_weight_vector)

    # candidate_tfidf_weight_matrix = np.array(candidate_tfidf_weight_matrix).reshape(len(candidate_replies), embedding_dim)

    # To tensor
    query_tfidf_weight_vector = torch.Tensor(query_tfidf_weight_vector)
    candidate_tfidf_weight_matrix = torch.Tensor(candidate_tfidf_weight_matrix)

    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    return cos(query_tfidf_weight_vector, candidate_tfidf_weight_matrix)

def get_avg_embedding_score(pre_trained_embedding, vector_query, matrix_candidate):

    avg_vector_query = np.mean(
        [pre_trained_embedding[id_query] for id_query in vector_query],
        axis=0
    ).reshape(1, -1)

    avg_matrix_candidate = np.array([
        np.mean([pre_trained_embedding[id_candidate] for id_candidate in vector_candidate], axis=0)
        for vector_candidate in matrix_candidate
    ])

    avg_vector_query = torch.Tensor(avg_vector_query)
    avg_matrix_candidate = torch.Tensor(avg_matrix_candidate)

    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    return cos(avg_vector_query, avg_matrix_candidate)



'''
Word Mover's Distance
'''
def get_wmd_score(vocab, pre_trained_embedding, vector_query, input_str, matrix_candidate, candidate_replies,  tfidf, stop_word_obj):

    from gensim.models import Word2Vec
    model = Word2Vec.load_word2vec_format()

    model.wv.wmdistance

    pass

