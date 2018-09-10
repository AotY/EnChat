# -*- coding:utf-8 -*-
from __future__ import division
from __future__ import print_function

import argparse


# from opennmt

def preprocess_opts(parser):
    # Data options
    group = parser.add_argument_group('Data Preprocess')

    group.add_argument('-train_corpus_path', nargs='+', required=True,
                       help="Path to the training source data")

    group.add_argument('-valid_corpus_path', nargs='+', required=True,
                       help="Path to the validation source data")

    group.add_argument('-test_corpus_path', nargs='+',
                       help="Path to the test source data")

    group.add_argument('-save_data', required=True,
                       help="Output file for the prepared data")

    group.add_argument('-max_shard_size', type=int, default=0,
                       help="""For text corpus of large volume, it will
                       be divided into shards of this size to preprocess.
                       If 0, the data will be handled as a whole. The unit
                       is in bytes. Optimal value should be multiples of
                       64 bytes.""")

    group.add_argument('-with_label', action='store_true', help='with label')

    # Dictionary options, for text corpus

    group = parser.add_argument_group('Vocab')
    group.add_argument('-vocab_path', default="",
                       help="""Path to an existing source vocabulary. Format:
                       one word per line.""")

    group.add_argument('-vocab_size', type=int, default=50000,
                       help="Size of the source vocabulary")

    group.add_argument('-words_min_frequency', type=int, default=0)

    group.add_argument('-share_vocab', action='store_true',
                       help="Share source and target vocabulary")

    group.add_argument('-vocab_we_save_path',
                       help="""Path to save word embedding of vocab,
                       building from pre-trained word embedding. 
                       (eg. word2vec, glove, or fasttext)
                       """)


    # Truncation options, for text corpus
    group = parser.add_argument_group('Pruning')
    group.add_argument('-seq_length', type=int, default=50,
                       help="Maximum source sequence length")
    group.add_argument('-seq_length_trunc', type=int, default=0,
                       help="Truncate source sequence length.")

    group.add_argument('-lower', action='store_true', help='lowercase data')


    # Data processing options
    group = parser.add_argument_group('Random')

    group.add_argument('-shuffle', type=int, default=1,
                       help="Shuffle data")

    group.add_argument('-seed', type=int, default=3435,
                       help="Random seed")

    group.add_argument('-stop_word_file',
                       default='./stopwords_en.txt',
                       help="For loding stop_words")


    # from previous vocab
    group.add_argument('-pre_word_vecs_path',
                       help="""If a valid path is specified, then this will load
                       pretrained word embeddings on the encoder side.
                       See README for specific formatting instructions.""")

    group.add_argument('-pre_word_vecs_type',
                       default='word2vec',
                       help="""pre-trained word embedding type. eg. word2vec, glove. 
                       """)

    group.add_argument('-pre_word_vecs_dim',
                       type=int,
                       default=300,
                       help="""pre-trained word embedding dim. 
                           """)

    group.add_argument('-binary',
                       type=bool,
                       default=False,
                       help="""pre-trained word embedding file type, binary(True) or txt(False). 
                           """)

    group.add_argument('-pre_trained_vocab_embedding_file',
                       help="""pre trained vocab embedding file 
                           """)

    # TFIDF
    group.add_argument('-vocab_tfidf',
                       help="""This object saves the idf value of all vocab words. 
                           """)
