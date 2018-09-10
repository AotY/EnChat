# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
from collections import Counter

'''
Get embedding score with tfidf weight
'''


def get_tfidf_embedding_score(vocab, pre_trained_embedding, input_str, candidate_replies, tfidf, stop_word_obj):
    # embedding dim
    embedding_dim = pre_trained_embedding.shape[1]

    query_words = stop_word_obj.remove_words(input_str.strip().replace('\t', ' '))
    query_len = len(query_words)

    # init
    query_tfidf_weights = np.array([round(1.0 / query_len, 5)] * query_len)
    vector_query_ids = [vocab.unkid] * query_len

    # freq
    query_words_counter = Counter(query_words)

    # conpute tfidf
    query_words_set = set(query_words)
    for word in query_words_set:
        print('query_word:    {}'.format(word))
        word_id = vocab.word2idx.get(word, vocab.unkid)

        if word_id == vocab.unkid:
            continue

        tf = query_words_counter[word] / query_len
        idf = tfidf.idf_dict.get(word, 0.0)
        word_tfidf = tf * idf

        print('word_tfidf:    {}'.format(word_tfidf))
        if word_tfidf != 0:
            for idx, word2 in enumerate(query_words):
                if word == word2:
                    query_tfidf_weights[idx] = word_tfidf
                    vector_query_ids[idx] = word_id

    # scalar
    query_min_weight = np.min(query_tfidf_weights)
    query_max_weight = np.max(query_tfidf_weights)
    if query_min_weight != query_max_weight:
        query_tfidf_weights = np.divide((query_tfidf_weights - query_min_weight),
                                        (query_max_weight - query_min_weight) * 1.0)
    else:
        query_tfidf_weights = np.divide(query_tfidf_weights, np.sum(query_tfidf_weights))

    query_tfidf_weight_vector = np.zeros(embedding_dim, dtype=np.float64)

    for id_query, weight in zip(vector_query_ids, query_tfidf_weights):
        query_tfidf_weight_vector = np.add(query_tfidf_weight_vector, pre_trained_embedding[id_query] * weight)

    candidate_tfidf_weight_matrix = []
    for candidate_replie in candidate_replies:
        candidate_words = stop_word_obj.remove_words(candidate_replie.strip().replace('\t', ' '))
        candidate_len = len(candidate_replie)
        candidate_words_counter = Counter(candidate_replie)

        candidate_tfidf_weights = np.array([round(1.0 / candidate_len, 5)] * candidate_len)
        vector_candidate_ids = [vocab.unkid] * candidate_len

        # conpute tfidf
        candidate_words_set = set(candidate_words)

        for word in candidate_words_set:
            word_id = vocab.word2idx.get(word, vocab.unkid)
            if word_id == vocab.unkid:
                continue

            tf = candidate_words_counter[word] / candidate_len
            idf = tfidf.idf_dict.get(word, 0.0)
            word_tfidf = tf * idf

            print('word_tfidf:  {}'.format(word_tfidf))
            if word_tfidf != 0:
                for idx, word2 in enumerate(candidate_words):
                    if word == word2:
                        candidate_tfidf_weights[idx] = word_tfidf
                        vector_candidate_ids[idx] = word_id

        # scalar
        candidate_min_weight = np.min(candidate_tfidf_weights)
        candidate_max_weight = np.max(candidate_tfidf_weights)

        if candidate_min_weight != candidate_max_weight:
            candidate_tfidf_weights = np.divide((candidate_tfidf_weights - candidate_min_weight),
                                                (candidate_max_weight - candidate_min_weight) * 1.0)
        else:
            candidate_tfidf_weights = np.divide(candidate_tfidf_weights, np.sum(candidate_tfidf_weights))

        candidate_tfidf_weight_vector = np.zeros(embedding_dim, dtype=np.float64)

        for id_can, weight in zip(vector_candidate_ids, candidate_tfidf_weights):
            candidate_tfidf_weight_vector += pre_trained_embedding[id_can] * weight

        candidate_tfidf_weight_matrix.append(candidate_tfidf_weight_vector)

    query_tfidf_weight_vector = query_tfidf_weight_vector.reshape(1, -1)
    candidate_tfidf_weight_matrix = np.array(candidate_tfidf_weight_matrix).reshape(len(candidate_replies),
                                                                                    embedding_dim)

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


def get_wmd_score(vocab, pre_trained_embedding, vector_query, input_str, matrix_candidate, candidate_replies, tfidf,
                  stop_word_obj):
    from gensim.models import Word2Vec
    model = Word2Vec.load_word2vec_format()

    model.wv.wmdistance

    pass
