# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
from collections import Counter


def get_tfidf_embedding_score(vocab, pre_trained_embedding, vector_query, matrix_candidate, tfidf, stop_word_obj, input_str, candidate_replies):

    embedding_dim = pre_trained_embedding.shape[1]

    query_tfidf_weight_vector = np.array([round(1./ embedding_dim, 5)] * embedding_dim)

    query_words = stop_word_obj.remove_words(input_str.strip().replace('\t', ' '))
    query_len = len(query_words)
    query_words_counter = Counter(query_words)

    # conpute tfidf
    for idx, word_id in enumerate(vector_query):
        word = vocab.idx2word[word_id]
        if word in query_words:
            word_tfidf = ( query_words_counter[word] / query_len) * tfidf.get(word, 0)
            if word_tfidf != 0:
                query_tfidf_weight_vector[idx] = word_tfidf

    query_tfidf_weight_vector = (query_tfidf_weight_vector - np.min(query_tfidf_weight_vector)) / (np.max(query_tfidf_weight_vector) - np.min(query_tfidf_weight_vector))

    candidate_tfidf_weight_matrix = []
    for candidate_replie, vector_candidate in zip(candidate_replies, matrix_candidate):
        candidate_tfidf_weight_vector = np.array([round(1. / embedding_dim, 5)] * embedding_dim)
        candidate_words = stop_word_obj.remove_words(input_str.strip().replace('\t', ' '))
        candidate_len = len(query_words)
        candidate_words_counter = Counter(query_words)

        # conpute tfidf
        for idx, word_id in enumerate(vector_candidate):
            word = vocab.idx2word[word_id]
            if word in query_words:
                word_tfidf = (candidate_words_counter[word] / query_len) * tfidf.get(word, 0)
                if word_tfidf != 0:
                    candidate_tfidf_weight_vector[idx] = word_tfidf
        candidate_tfidf_weight_matrix.append(candidate_tfidf_weight_vector)

    candidate_tfidf_weight_matrix = np.array(candidate_tfidf_weight_matrix).reshape(len(candidate_replies, embedding_dim))

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


