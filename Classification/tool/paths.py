# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
# local or remote
flag = 'local'
# flag = 'remote'

if flag == 'local':
    train_label_file = '/Users/LeonTao/PycharmProjects/AskQA/data/test.label.txt'
    train_post_file = '/Users/LeonTao/PycharmProjects/AskQA/data/test.post.txt'
    train_response_file = '/Users/LeonTao/PycharmProjects/AskQA/data/test.response.txt'

    test_label_file = '/Users/LeonTao/PycharmProjects/AskQA/data/valid.label.txt'
    test_post_file = '/Users/LeonTao/PycharmProjects/AskQA/data/valid.post.txt'
    test_response_file = '/Users/LeonTao/PycharmProjects/AskQA/data/valid.response.txt'

    vocab_file = '/Users/LeonTao/PycharmProjects/AskQA/data/askqa.vocab.txt'

    word2vec_file = '/Users/LeonTao/NLP/Corpos/GoogleNews-vectors-negative300.bin'

elif flag == 'remote':
    train_label_file = '/home/Research/askqa/train.label.txt'
    train_post_file = '/home/Research/askqa/test.post.txt'
    train_response_file = '/home/Research/askqa/test.response.txt'

    test_label_file = '/home/Research/askqa/test.label.txt'
    test_post_file = '/home/Research/askqa/test.post.txt'
    test_response_file = '/home/Research/askqa/test.response.txt'

    valid_label_file = '/home/Research/askqa/valid.label.txt'
    valid_post_file = '/home/Research/askqa/valid.label.txt'
    valid_response_file = '/home/Research/askqa/valid.label.txt'

    vocab_file = '/home/taoqing/askqa/askqa.vocab.txt'

    word2vec_file = '/home/taoqing/Research/data/GoogleNews-vectors-negative300.bin'



if __name__ == '__main__':
    print('train_label_file: {} \ntest_label_file: {} \n'.format(train_label_file, test_label_file))