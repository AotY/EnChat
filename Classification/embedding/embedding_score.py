# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
from collections import Counter
import gensim

'''
Get embedding score with tfidf weight
'''

'''
def get_tfidf_embedding_score(vocab, pre_trained_embedding, input_str, candidate_replies, tfidf, stop_word_obj, lower=None):
    # embedding dim
    embedding_dim = pre_trained_embedding.shape[1]

    query_words = stop_word_obj.remove_words(input_str.strip().replace('\t', ' '))
    # to lower
    if lower:
        query_words = [word.lower() for word in query_words]

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
    for candidate_reply in candidate_replies:
        candidate_words = stop_word_obj.remove_words(candidate_reply.strip().replace('\t', ' '))

        # to lower
        # candidate_words = [word.lower() for word in candidate_words]

        candidate_len = len(candidate_reply)
        candidate_words_counter = Counter(candidate_reply)

        candidate_tfidf_weights = np.array([round(1.0 / candidate_len, 5)] * candidate_len)
        vector_candidate_ids = [vocab.unkid] * candidate_len

        # conpute tfidf
        candidate_words_set = set(candidate_words)

        for word in candidate_words_set:
            print ('word: {}'.format(word))
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
'''

def get_tfidf_embedding_score(vocab, gensim_model, tfidf, input_str, candidate_replies, stop_word_obj, lower=None):

    query_words = stop_word_obj.remove_words(input_str.strip().replace('\t', ' '))
    query_len = len(query_words)
    # to lower
    if lower:
        query_words = [word.lower() for word in query_words]

    # init
    query_tfidf_weights = np.array([round(1.0 / query_len, 5)] * query_len)

    # freq
    query_words_counter = Counter(query_words)

    query_words_embedding = []
    for idx, word in enumerate(query_words):
        try:
            word_embedding = gensim_model.wv[word]
        except KeyError:
            word_embedding = gensim_model.wv[vocab.unk]
        query_words_embedding.append(word_embedding)

        word_tfidf = (query_words_counter[word] / query_len) * (tfidf.idf_dict.get(word, 0.0))
        print('query_word: {}, word_tfidf:    {}'.format(word, word_tfidf))

        query_tfidf_weights[idx] = word_tfidf

    # scalar
    query_max_weight = np.max(query_tfidf_weights)
    query_min_weight = np.min(query_tfidf_weights)
    if query_min_weight != query_max_weight:
        query_tfidf_weights = np.divide((query_tfidf_weights - query_min_weight),
                                        (query_max_weight - query_min_weight) * 1.0)
    else:
        query_tfidf_weights = np.divide(query_tfidf_weights, np.sum(query_tfidf_weights))

    query_tfidf_weight_vector = np.zeros_like(word_embedding, dtype=np.float64)

    for word_embedding, weight in zip(query_words_embedding, query_tfidf_weights):
        query_tfidf_weight_vector = np.add(query_tfidf_weight_vector, word_embedding * weight)

    tfidf_matrix_candidate = []
    for candidate_reply in candidate_replies:
        candidate_words = stop_word_obj.remove_words(candidate_reply.strip().replace('\t', ' '))

        # to lower
        if lower:
            candidate_words = [word.lower() for word in candidate_words]

        candidate_len = len(candidate_words)

        # init
        candidate_tfidf_weights = np.array([round(1.0 / candidate_len, 5)] * candidate_len)

        # freq
        candidate_words_counter = Counter(candidate_words)

        candidate_words_embedding = []
        for idx, word in enumerate(candidate_words):
            try:
                word_embedding = gensim_model.wv[word]
            except KeyError:
                word_embedding = gensim_model.wv[vocab.unk]
            candidate_words_embedding.append(word_embedding)

            word_tfidf = (candidate_words_counter[word] / candidate_len) * (tfidf.idf_dict.get(word, 0.0))
            print('candidate_word: {}, word_tfidf:    {}'.format(word, word_tfidf))

            candidate_tfidf_weights[idx] = word_tfidf

        # scalar
        candidate_max_weight = np.max(candidate_tfidf_weights)
        candidate_min_weight = np.min(candidate_tfidf_weights)
        if candidate_min_weight != candidate_max_weight:
            candidate_tfidf_weights = np.divide((candidate_tfidf_weights - candidate_min_weight),
                                            (candidate_max_weight - candidate_min_weight) * 1.0)
        else:
            candidate_tfidf_weights = np.divide(candidate_tfidf_weights, np.sum(candidate_tfidf_weights))

        candidate_tfidf_weight_vector = np.zeros_like(word_embedding, dtype=np.float64)

        for word_embedding, weight in zip(candidate_words_embedding, candidate_tfidf_weights):
            candidate_tfidf_weight_vector = np.add(candidate_tfidf_weight_vector, word_embedding * weight)

        tfidf_matrix_candidate.append(candidate_tfidf_weight_vector)


    tfidf_matrix_candidate = np.array(tfidf_matrix_candidate)

    score_vector = gensim_model.cosine_similarities(query_tfidf_weight_vector, tfidf_matrix_candidate)

    # normalization
    max_score = np.max(score_vector)
    min_score = np.min(score_vector)
    score_vector = np.divide((score_vector - min_score), (max_score - min_score) * 1.0)

    return score_vector


'''
Average:
An utterance representation can be obtained by averaging the embeddings of all the words in that utterance, of which the cosine similarity gives the Average metric
'''
def get_avg_embedding_score(vocab, gensim_model, input_str, candidate_replies, stop_word_obj, lower=None):

    query_words = stop_word_obj.remove_words(input_str.strip().replace('\t', ' '))

    # to lower
    if lower:
        query_words = [word.lower() for word in query_words]

    query_words_embedding = []
    for word in query_words:
        print('query_word: ', word)
        try:
            word_embedding = gensim_model.wv[word]
        except KeyError:
            word_embedding = gensim_model.wv[vocab.unk]
        query_words_embedding.append(word_embedding)

        print('word: {}, word_embedding: {}'.format(word, word_embedding))
    avg_vector_query = np.array(query_words_embedding).mean(axis=0)

    avg_matrix_candidate = []
    for candidate_reply in candidate_replies:
        candidate_words = stop_word_obj.remove_words(candidate_reply.strip().replace('\t', ' '))

        # to lower
        if lower:
            candidate_words = [word.lower() for word in candidate_words]

        candidate_words_embedding = []
        for word in candidate_words:
            try:
                word_embedding = gensim_model.wv[word]
            except KeyError:
                word_embedding = gensim_model.wv[vocab.unk]
            candidate_words_embedding.append(word_embedding)

        avg_vector_candidate = np.array(candidate_words_embedding).mean(axis=0)

        avg_matrix_candidate.append(avg_vector_candidate)

    avg_matrix_candidate = np.array(avg_matrix_candidate)

    score_vector = gensim_model.cosine_similarities(avg_vector_query, avg_matrix_candidate)

    # normalization
    max_score = np.max(score_vector)
    min_score = np.min(score_vector)
    score_vector = np.divide((score_vector - min_score), (max_score - min_score) * 1.0)

    return score_vector

'''
Extreme:
Achieve an utterance representation by taking the largest extreme values among the embedding vectors of all the words it contains
'''
def get_extreme_embedding_score(vocab, gensim_model, input_str, candidate_replies, stop_word_obj, lower=None):
    query_words = stop_word_obj.remove_words(input_str.strip().replace('\t', ' '))
    # to lower
    if lower:
        query_words = [word.lower() for word in query_words]

    query_words_embedding = []
    for word in query_words:
        try:
            word_embedding = gensim_model.wv[word]
        except KeyError:
            word_embedding = gensim_model.wv[vocab.unk]
        query_words_embedding.append(word_embedding)

    extreme_vector_query = np.array(query_words_embedding).max(axis=0)

    extreme_matrix_candidate = []
    for candidate_reply in candidate_replies:
        candidate_words = stop_word_obj.remove_words(candidate_reply.strip().replace('\t', ' '))

        # to lower
        if lower:
            candidate_words = [word.lower() for word in candidate_words]

        candidate_words_embedding = []
        for word in candidate_words:
            try:
                word_embedding = gensim_model.wv[word]
            except KeyError:
                word_embedding = gensim_model.wv[vocab.unk]

            candidate_words_embedding.append(word_embedding)

        extreme_vector_candidate = np.array(candidate_words_embedding).max(axis=0)
        extreme_matrix_candidate.append(extreme_vector_candidate)

    extreme_matrix_candidate = np.array(extreme_matrix_candidate)

    score_vector = gensim_model.cosine_similarities(extreme_vector_query, extreme_matrix_candidate)

    # normalization
    max_score = np.max(score_vector)
    min_score = np.min(score_vector)
    score_vector = np.divide((score_vector - min_score), (max_score - min_score) * 1.0)

    return score_vector

'''
Greedy:
Greedily match words in two given utterances based on the cosine similarities of their embeddings, and to average the obtained scores
'''
def get_greedy_embedding_score(gensim_model, input_str, candidate_replies, stop_word_obj, lower=None):

    query_words = stop_word_obj.remove_words(input_str.strip().replace('\t', ' '))

    # to lower
    if lower:
        query_words = [word.lower() for word in query_words]

    score_vector = []
    for candidate_reply in candidate_replies:
        candidate_words = stop_word_obj.remove_words(candidate_reply.strip().replace('\t', ' '))

        # to lower
        if lower:
            candidate_words = [word.lower() for word in candidate_words]

        max_scores = []
        for query_word in query_words:
            max_score = 0.0
            for candidate_word in candidate_words:
                try:
                    score = gensim_model.wv.similarity(query_word, candidate_word)
                except KeyError:
                    score = -1.0

                max_score = max(score, max_score)

            max_scores.append(max_score)

        score_vector.append(np.mean(max_scores))


    # normalization
    max_score = np.max(score_vector)
    min_score = np.min(score_vector)
    score_vector = np.divide((score_vector - min_score), (max_score - min_score) * 1.0)

    return score_vector


'''
Optimal Matching
TODO
'''

'''
Word Mover's Distance
'''

def get_wmd_embedding_score(gensim_model, input_str, candidate_replies, stop_word_obj, lower=None):

    distance_vector = np.random.rand(len(candidate_replies))

    query_words = stop_word_obj.remove_words(input_str.strip().replace('\t', ' '))
    # to lower
    if lower:
        query_words = [word.lower() for word in query_words]

    for idx, candidate_reply in enumerate(candidate_replies):
        candidate_words = stop_word_obj.remove_words(candidate_reply.strip().replace('\t', ' '))
        # to lower
        if lower:
            candidate_words = [word.lower() for word in candidate_words]

        distance_vector[idx] = gensim_model.wmdistance(query_words, candidate_words)

    # distance to score
    score_vector = np.max(distance_vector) - distance_vector

    # normalization
    max_score = np.max(score_vector)
    min_score = np.min(score_vector)
    score_vector = np.divide((score_vector - min_score), (max_score - min_score) * 1.0)

    return score_vector



'''
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

