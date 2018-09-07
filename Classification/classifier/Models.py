"""
This file is for several classification architectures, 
which are universal model for text classification tasks.
"""
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
import math

################################## Single Encoder #####################################
class SingleEncoder(nn.Module):
    """ This is a general architecture for classification tasks 
        that only have one input.
        Generally, such frameworks just contains one Encoder and
        a score function.

        score function options:
          Multi-layer Perceptron (MLP)
          Logistic Regression (LR) (multi-classification)
          Dual, refer to the Dual model 
          Bilinear or Linear Transformation (obj "nn.ilinear")
          ... ... 
    """
    def __init__(self, encoder, 
                       input_size, hidden_size,
                       output_size, dropout=0.0,
                       score_fn_type='MLP'):
        """for a simple, generic encoder + a score function.
        Args:
            encoder (:obj:`EncoderBase`): an encoder object
            input_size: the output_size of encoder
            hidden_size: the dimension of the score function
            output_size: the class number of targets
        """
        super(SingleEncoder, self).__init__()
        self.encoder = encoder
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout = dropout ### for extension 
        self.score_fn_type = score_fn_type
        self.cls_arch = 'SingleEncoder'

        # define score function, now only MLP 
        # for classification, 
        self.score_fn, self.bias = self.build_classifier()

    def build_classifier(self):
        """
        """
        if self.score_fn_type == 'MLP':
            return (nn.Sequential(
                       nn.Linear(self.input_size, self.hidden_size),
                       nn.Tanh(),
                       nn.Linear(self.hidden_size, self.output_size)),
                       None
                    )
        elif self.score_fn_type == 'LR':
            weight = nn.Parameter(torch.Tensor(self.input_size))
            bias = nn.Parameter(torch.Tensor(1))
            # initialization
            stdv = 1. / math.sqrt(weight.size(0))
            weight.data.uniform_(-stdv, stdv)
            bias.data.uniform_(-stdv, stdv)
            return weight, bias
        elif self.score_fn_type == 'DUAL':
            return (nn.Linear(self.input_size, self.hidden_size, bias=False),
                    None)
        else:
            raise ValueError("{} is not valid, ".format(self.score_fn_type))

    def forward(self, src, lengths, state=None):
        """Forward propagate a `src` and `tgt` pair for training.
        Possible initialized with a beginning decoder state.

        Args:
            src (:obj:`Tensor`):
                a source sequence passed to encoder.
                typically for inputs this will be a padded :obj:`LongTensor`
                of size `[len x batch]`. however, may be an
                image or other generic input depending on encoder.
            lengths(:obj:`LongTensor`): the src lengths, pre-padding `[batch]`.
            state (:obj:`DecoderState`, optional): encoder initial state

        Returns:
                 * decoder output `[batch x output_size]`
                 * final decoder state
        """
        raise NotImplementedError

############################ Single Input ###########################
class SingleArch(SingleEncoder):
    """ This is a general architecture for classification tasks 
        that only have one input.
        Generally, such frameworks just contains one Encoder and
        a score function.

        score function options:
          Multi-layer Perceptron (MLP)
          Logistic Regression (LR)
          ... ... 
    """
    def forward(self, src, lengths, state=None):
        """Forward propagate a `src` and `tgt` pair for training.
        Possible initialized with a beginning decoder state.

        Args:
            src (:obj:`Tensor`):
                a source sequence passed to encoder.
                typically for inputs this will be a padded :obj:`LongTensor`
                of size `[len x batch]`. however, may be an
                image or other generic input depending on encoder.
            lengths(:obj:`LongTensor`): the src lengths, pre-padding `[batch]`.
            state (:obj:`DecoderState`, optional): encoder initial state

        Returns:
                 * decoder output `[batch x output_size]`, logists
                 * final decoder state
        """
        _, batch_size = src.size()

        enc_final, memory_bank = self.encoder(src, lengths=lengths)
        
        if not isinstance(enc_final, tuple):
            encoder_hidden = (enc_final, )
        # 
        encoder_hidden = encoder_hidden[0].contiguous().view(batch_size, -1)
        if self.score_fn_type == 'MLP':
            outputs = self.score_fn(encoder_hidden)
            # outputs = F.softmax(outputs)
        elif self.score_fn_type == 'LR':
            outputs = torch.mv(encoder_hidden, self.score_fn) + self.bias
            # outputs = F.sigmoid(outputs)
        else:
            raise ValueError("{} is not valid for SingleArch, ".format(self.score_fn_type))
        # outputs = torch.clamp(outputs, 1e-7, 1.0-1e-7)

        return outputs, enc_final

################################ Double Input ###########################

class DoubleArch(SingleEncoder):
    """This is a general architecture for classification tasks 
        that have two inputs (for example, one is a query, 
        the other is the corresponding response.).
        Generally, such frameworks just contains one Encoder and
        a score function.

        score function options:
          Multi-layer Perceptron (MLP)
          Logistic Regression (LR)
          ... ... 
        : without any extended feature, like bilinear feature, dot feature 
    """
    def __init__(self, encoder, 
                       input_size, hidden_size,
                       output_size, dropout=0.0, 
                       score_fn_type='DUAL'):
        """
        """
        if score_fn_type in ['MLP', 'LR']:
            input_size = 2 * input_size
        super(DoubleArch, self).__init__(encoder, 
                       input_size, hidden_size,
                       output_size, dropout=dropout,
                       score_fn_type=score_fn_type)
        # 
        self.cls_arch = 'DualArch' 
    
    # def build_classifier(self):
    #     """
    #     """
    #     raise NotImplementedError 

    def forward(self, q_src, q_lengths, r_src, r_lengths, state=None):
        """
        """
        _, batch_size = q_src.size()

        q_enc_final, q_memory_bank = self.encoder(q_src, lengths=q_lengths)
        r_enc_final, r_memory_bank = self.encoder(r_src, lengths=r_lengths)
        if not isinstance(q_enc_final, tuple):
            q_encoder_hidden = (q_enc_final, )
            r_encoder_hidden = (r_enc_final, )
        else:
            q_encoder_hidden = q_enc_final
            r_encoder_hidden = r_enc_final
        # 
        q_encoder_hidden = q_encoder_hidden[0].contiguous().view(batch_size, -1)
        r_encoder_hidden = r_encoder_hidden[0].contiguous().view(batch_size, -1)
        
        if self.score_fn_type == 'DUAL':
            q_transform = self.score_fn(q_encoder_hidden).unsqueeze(1)
            outputs = torch.bmm(q_transform, r_encoder_hidden.unsqueeze(-1))
        else:
            qr_con = torch.cat((q_encoder_hidden, r_encoder_hidden), dim=-1)
            if self.score_fn_type == 'MLP':
                outputs = self.score_fn(qr_con)
            elif self.score_fn_type == 'LR':
                outputs = torch.mv(qr_con, self.score_fn) + self.bias
            else:
                raise ValueError("{} is not valid.".format(self.score_fn_type))
        #if self.output_size == 1:
        #    outputs = F.sigmoid(outputs)
        #else:
        #    outputs = F.softmax(outputs)
        #outputs = torch.clamp(outputs, 1e-7, 1.0-1e-7)

        return outputs.squeeze(), q_enc_final


class ExtendDoubleArch(SingleEncoder):
    """This is a general architecture for classification tasks 
        that have two inputs (for example, one is a query, 
        the other is the corresponding response.).
        Generally, such frameworks just contains one Encoder and
        a score function.

        score function options:
          Multi-layer Perceptron (MLP)
          Logistic Regression (LR)
          ... ... 
        : with any extended features, 
          like bilinear feature, 
               element-wise dot or substract feature,
               inner-product,
               etc. 
    """
    def __init__(self, encoder, 
                       input_size, hidden_size,
                       output_size, dropout=0.0, 
                       score_fn_type='MLP',
                       bilinear_flag=False,
                       dot_flag=False,
                       substract_flag=False,
                       inner_prod_flag=False):
        """
        """
        # update the input size of the score function
        new_input_size = 0 #input_size
        if dot_flag:
            new_input_size += input_size
        if substract_flag:
            new_input_size += input_size
        if inner_prod_flag:
            new_input_size += 1
        if bilinear_flag:
            new_input_size += hidden_size

        super(ExtendDoubleArch, self).__init__(encoder, 
                       new_input_size, hidden_size,
                       output_size, dropout=dropout,
                       score_fn_type=score_fn_type)

        self.bilinear_flag = bilinear_flag
        self.dot_flag = dot_flag
        self.substract_flag = substract_flag
        self.inner_prod_flag = inner_prod_flag

        self.cls_arch = 'ExtendDoubleArch'
        if bilinear_flag:
            self.bilinear = nn.Bilinear(input_size, 
                         input_size, self.hidden_size) # ???

    def forward(self, q_src, q_lengths, r_src, r_lengths, state=None):
        """Forward propagate a `src` and `tgt` pair for training.
        Possible initialized with a beginning decoder state.

        Args:
            src (:obj:`Tensor`):
                a source sequence passed to encoder.
                typically for inputs this will be a padded :obj:`LongTensor`
                of size `[len x batch]`. however, may be an
                image or other generic input depending on encoder.
            tgt (:obj:`LongTensor`):
                 a target sequence of size `[tgt_len x batch]`.
            lengths(:obj:`LongTensor`): the src lengths, pre-padding `[batch]`.
            state (:obj:`DecoderState`, optional): encoder initial state
        Returns:
                 * decoder output `[batch x output_size]`
                 * final decoder state
        """
        _, batch_size = q_src.size()
        # encoder q and r
        q_enc_final, q_memory_bank = self.encoder(q_src, lengths=q_lengths)
        r_enc_final, r_memory_bank = self.encoder(r_src, lengths=r_lengths)
        if not isinstance(q_enc_final, tuple):
            q_encoder_hidden = (q_enc_final, )
            r_encoder_hidden = (r_enc_final, )
        else:
            q_encoder_hidden = q_enc_final
            r_encoder_hidden = r_enc_final
        q_encoder_hidden = q_encoder_hidden[0].contiguous().view(batch_size, -1)
        r_encoder_hidden = r_encoder_hidden[0].contiguous().view(batch_size, -1)

        # get extended features 
        extended_feats = []
        if self.substract_flag:
            substract_feats = q_encoder_hidden - r_encoder_hidden
            extended_feats.append(substract_feats)
        if self.dot_flag:
            dot_feats = torch.mul(q_encoder_hidden, r_encoder_hidden)
            extended_feats.append(dot_feats)
        if self.bilinear_flag:
            bilinear_feats = self.bilinear(q_encoder_hidden, r_encoder_hidden)
            extended_feats.append(bilinear_feats)
        if self.inner_prod_flag:
            inner_prob = torch.bmm(q_encoder_hidden.unsqueeze(1),
                                   r_encoder_hidden.unsqueeze(2))
            extended_feats.append(inner_prob.squeeze(-1))

        # concatenate features 
        combine_feats = torch.cat(extended_feats, dim=-1)
        # print(combine_feats.size(), self.input_size)
        if self.score_fn_type == 'MLP':
            outputs = self.score_fn(combine_feats)
        elif self.score_fn_type == 'LR':
            outputs = torch.mv(combine_feats, self.score_fn) + self.bias
        else:
            raise ValueError("{} is not valid for SingleArch, ".format(self.score_fn_type))
        #if self.output_size == 1:
        #    outputs = F.sigmoid(outputs)
        #else:
        #    outputs = F.softmax(outputs)
        #outputs = torch.clamp(outputs, 1e-7, 1.0-1e-7)

        return outputs.squeeze(), q_enc_final

########## Other classifier for q-r pairs

################################ Pairwise or Listwise Ranker #######################
class PairwiseRanker(nn.Module):
    """
    Core trainable object in OpenNMT. Implements a trainable interface
    for a simple, generic encoder + decoder model.

    Args:
      encoder (:obj:`EncoderBase`): an encoder object
      classifier, MLP or other type 
    """
    def __init__(self, encoder, input_size, hidden_size, 
                       output_size, score_fn_type="MLP"):
        super(PairCls, self).__init__()
        self.encoder = encoder
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.ranker_arch = 'pairwise'
        self.score_fn_type = score_fn_type

        # define score function
        self.score_fn, self.bias = self.build_classifier()

    def build_classifier(self):
        """
        """
        if self.score_fn_type == 'MLP':
            return (nn.Sequential(
                       nn.Linear(self.input_size, self.hidden_size),
                       nn.Tanh(),
                       nn.Linear(self.hidden_size, self.output_size)),
                       None
                    )
        elif self.score_fn_type == 'LR':
            weight = nn.Parameter(torch.Tensor(self.input_size))
            bias = nn.Parameter(torch.Tensor(1))
            # initialization
            stdv = 1. / math.sqrt(weight.size(0))
            weight.data.uniform_(-stdv, stdv)
            bias.data.uniform_(-stdv, stdv)
            return weight, bias
        elif self.score_fn_type == 'DUAL':
            return (nn.Linear(self.input_size, self.hidden_size, bias=False),
                    None)
        else:
            raise ValueError("{} is not valid, ".format(self.score_fn_type))

    def forward(self, q_src, q_lengths, pos_src, pos_lengths, 
                      neg_src, neg_lengths, state=None):
        """Forward propagate a `src` and `tgt` pair for training.
        Possible initialized with a beginning decoder state.

        Args:
            src (:obj:`Tensor`):
                a source sequence passed to encoder.
                typically for inputs this will be a padded :obj:`LongTensor`
                of size `[len x batch]`. however, may be an
                image or other generic input depending on encoder.
            tgt (:obj:`LongTensor`):
                 a target sequence of size `[tgt_len x batch]`.
            lengths(:obj:`LongTensor`): the src lengths, pre-padding `[batch]`.
            state (:obj:`DecoderState`, optional): encoder initial state
        Returns:
                 * decoder output `[batch x output_size]`
                 * final decoder state
        """
        _, batch_size = q_src.size()
        # embedding queries and responses.
        q_enc_final, q_memory_bank = self.encoder(q_src, lengths=q_lengths)
        pos_enc_final, pos_memory_bank = self.encoder(pos_src, lengths=pos_lengths)
        neg_enc_final, neg_memory_bank = self.encoder(neg_src, lengths=neg_lengths)
        if not isinstance(q_enc_final, tuple):
            q_encoder_hidden = (q_enc_final, )
            pos_encoder_hidden = (pos_enc_final, )
            neg_encoder_hidden = (neg_enc_final, )
        else:
            q_encoder_hidden = q_enc_final
            pos_encoder_hidden = pos_enc_final
            neg_encoder_hidden = neg_enc_final
        # 
        q_encoder_hidden = q_encoder_hidden[0].contiguous().view(batch_size, -1)
        pos_encoder_hidden = pos_encoder_hidden[0].contiguous().view(batch_size, -1)
        neg_encoder_hidden = neg_encoder_hidden[0].contiguous().view(batch_size, -1)
        # compute the relevance between queries and responses
        if self.score_fn_type == 'DUAL':
            q_transform = self.score_fn(q_encoder_hidden).unsqueeze(1)
            pos_outputs = torch.bmm(q_transform, pos_encoder_hidden.unsqueeze(-1))
            neg_outputs = torch.bmm(q_transform, neg_encoder_hidden.unsqueeze(-1))
        else:
            q_pos_con = torch.cat((q_encoder_hidden, pos_encoder_hidden), dim=-1)
            q_neg_con = torch.cat((q_encoder_hidden, neg_encoder_hidden), dim=-1)
            if self.score_fn_type == 'MLP':
                pos_outputs = self.score_fn(q_pos_con)
                neg_outputs = self.score_fn(q_neg_con)
            elif self.score_fn_type == 'LR':
                pos_outputs = torch.mv(q_pos_con, self.score_fn) + self.bias
                neg_outputs = torch.mv(q_neg_con, self.score_fn) + self.bias
            else:
                raise ValueError("{} is not valid.".format(self.score_fn_type))
        # 
        assert self.output_size == 1

        pos_outputs = torch.clamp(F.sigmoid(pos_outputs), 1e-7, 1.0-1e-7)
        neg_outputs = torch.clamp(F.sigmoid(neg_outputs), 1e-7, 1.0-1e-7)

        return pos_outputs, neg_outputs, q_enc_final
