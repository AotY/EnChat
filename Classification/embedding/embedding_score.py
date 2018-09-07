# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function

import numpy as np
import torch


def get_embedding_score(pre_trained_embedding, vector_query, matrix_candidate, type='avg'):

    avg_vector_query = np.mean(
                            [pre_trained_embedding[id_query] for id_query in vector_query],
                            axis=1
                            ).reshape(1, -1)

    avg_matrix_candidate = np.array([
            np.mean([
                pre_trained_embedding[id_candidate] for id_candidate in vector_candidate], axis=1
            )
            for vector_candidate in matrix_candidate]
    )

    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    return cos(avg_vector_query, avg_matrix_candidate)




