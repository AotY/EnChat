from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack

from utils import aeq, rnn_factory


class EncoderBase(nn.Module):
    """
    Base encoder class. Specifies the interface used by different encoder types
    and required by :obj:`modules.Encoder`.
    """
    def _check_args(self, input, lengths=None, hidden=None):
        s_len, n_batch = input.size()
        if lengths is not None:
            n_batch_, = lengths.size()
            aeq(n_batch, n_batch_)

    def forward(self, src, lengths=None, encoder_state=None):
        """
        Args:
            src (:obj:`LongTensor`):
               padded sequences of sparse indices `[src_len x batch]`
            lengths (:obj:`LongTensor`): length of each sequence `[batch]`
            encoder_state (rnn-class specific):
               initial encoder_state state.

        Returns:
            (tuple of :obj:`FloatTensor`, :obj:`FloatTensor`):
                * final encoder state, used to initialize decoder
                * memory bank for attention, `[src_len x batch x hidden]`
        """
        raise NotImplementedError

class MeanEncoder(EncoderBase):
    """A trivial non-recurrent encoder. Simply applies mean pooling.

    Args:
       num_layers (int): number of replicated layers
       embeddings (:obj:`onmt.modules.Embeddings`): embedding module to use
    """
    def __init__(self, num_layers, embeddings):
        super(MeanEncoder, self).__init__()
        self.num_layers = num_layers
        self.embeddings = embeddings
        self.hidden_size = embeddings.embedding_size

        self.encoder_type = 'mean'

    def forward(self, src, lengths=None, encoder_state=None):
        "See :obj:`EncoderBase.forward()`"
        """There still exists a problem about lengths.
           if length is not None, the mean should take it into account.
           But now, no such operation.
        """
        self._check_args(src, lengths, encoder_state)

        emb = self.embeddings(src)
        s_len, batch, emb_dim = emb.size()
        # calculating the embedding mean according to lengths
        if lengths is not None:
            lengths_exp = lengths.expand(emb_dim, batch).transpose(0, 1)
            mean_temp = emb.sum(0) / lengths_exp
            mean = mean_temp.expand(self.num_layers, batch, emb_dim)
        else:
            mean = emb.mean(0).expand(self.num_layers, batch, emb_dim)
        memory_bank = emb
        encoder_final = (mean, mean)
        return encoder_final, memory_bank

class RNNEncoder(EncoderBase):
    """ A generic recurrent neural network encoder.

    Args:
       rnn_type (:obj:`str`):
          style of recurrent unit to use, one of [RNN, LSTM, GRU, SRU]
       bidirectional (bool) : use a bidirectional RNN
       num_layers (int) : number of stacked layers
       hidden_size (int) : hidden size of each layer
       dropout (float) : dropout value for :obj:`nn.Dropout`
       embeddings (:obj:`onmt.modules.Embeddings`): embedding module to use

    """
    def __init__(self, rnn_type, bidirectional, num_layers,
                 hidden_size, dropout=0, embeddings=None):
        super(RNNEncoder, self).__init__()
        assert embeddings is not None

        num_directions = 2 if bidirectional else 1
        assert hidden_size % num_directions == 0
        hidden_size = hidden_size // num_directions
        self.embeddings = embeddings

        self.rnn = rnn_factory(rnn_type,
                        input_size=embeddings.embedding_size,
                        hidden_size=hidden_size,
                        num_layers=num_layers,
                        dropout=dropout,
                        bidirectional=bidirectional)
        self.encoder_type = 'rnn'
        self.rnn_type = rnn_type

    def forward(self, src, lengths=None, encoder_state=None):
        """
        Args:
            src (:obj:`LongTensor`):
            padded sequences of sparse indices `[src_len x batch]`
            lengths (:obj:`LongTensor`): length of each sequence `[batch]`
            encoder_state (rnn-class specific):
               initial encoder_state state.

        Returns:
            (tuple of :obj:`FloatTensor`, :obj:`FloatTensor`):
                * final encoder state, used to initialize decoder
                * memory bank for attention, `[src_len x batch x hidden]`
        """
        self._check_args(src, lengths, encoder_state)

        # rank the sequences according to their lengths 
        input_length = Variable(lengths)
        if src.is_cuda:
            input_length = input_length.cuda()
        new_input_length, length_inds = torch.sort(input_length, descending=True)
        new_src = torch.index_select(src, 1, length_inds)
        src, lengths = (new_src, new_input_length.data)

        emb = self.embeddings(src)
        s_len, batch, emb_dim = emb.size()

        packed_emb = emb
        if lengths is not None:
            # Lengths data is wrapped inside a Variable.
            lengths = lengths.view(-1).tolist()
            packed_emb = nn.utils.rnn.pack_padded_sequence(packed_emb, lengths)

        memory_bank, encoder_final = self.rnn(packed_emb, encoder_state)

        if lengths is not None:
            memory_bank = nn.utils.rnn.pad_packed_sequence(memory_bank)[0]
        # map to input order 
        _, out_order = torch.sort(length_inds)
        if isinstance(encoder_final, tuple):
            out_memory_bank, out_encode_final = (torch.index_select(memory_bank, 1, out_order),
                                     (torch.index_select(encoder_final[0], 1, out_order), 
                                      torch.index_select(encoder_final[1], 1, out_order)))
        else:
            out_memory_bank, out_encode_final = (torch.index_select(memory_bank, 1, out_order),
                                     torch.index_select(encoder_final, 1, out_order))
        return (out_encode_final, out_memory_bank)
# 
class CNNEncoder(EncoderBase):
    """
        a simple CNN Encoder
    """
    def __init__(self, hidden_size, num_layers=1, 
                    filter_num=64, filter_sizes=[1, 2, 3, 4], 
                    dropout=0, embeddings=None):
        """
            embed_size,
            hidden_size,
            embedder,
            filter=num,
            filter_sizes,
            dropout,
        """
        super(CNNEncoder, self).__init__()
        assert embeddings is not None 
        self.input_size=embeddings.embedding_size
        self.hidden_size = hidden_size
        self.filter_num = filter_num
        self.filter_sizes = filter_sizes
        self.dropout_p = dropout
        self.embeddings = embeddings

        self.encoder_type = 'simple_cnn'
        # input shape of Conv2d, (batch_size, 1, max_seqlen, embedding_size)
        self.convs = nn.ModuleList( [ nn.Conv2d(1, filter_num, (filter_size, self.input_size), 
                                       stride=(1, 1), padding=(filter_size//2, 0), dilation=1, bias=False) 
                                       for filter_size in filter_sizes ])
        if self.dropout_p > 0.:
            self.dropout = nn.Dropout(self.dropout_p)

        # output layer
        self.pool_feat_num = self.filter_num*len(self.filter_sizes)
        self.output_layer = nn.Linear(self.pool_feat_num, self.hidden_size)
        self.tanh = nn.Tanh()

    def forward(self, src, lengths=None, encoder_state=None):
        '''
            input, seq_len * batch_size, time-step control
                 ,
            output, batch_size * (len(filter_sizes) X filter_num)
        '''
        #s_len, batch, emb_dim = emb.size()
        embeded = self.embeddings( src.transpose(0, 1) ) # batch_size * seq_len * hidden
        embeded = embeded.unsqueeze(1)  # batch_size * 1 * seq_len * hidden
        s_len, _, batch, emb_dim = embeded.size()
        conv_feats = [F.relu(conv(embeded)).squeeze(3) for conv in self.convs] 
        # [batch_size * filter_num * seq_len] * len(filter_sizes)
        pooled_feats = [F.max_pool1d(feat, feat.size(2)).squeeze(2) for feat in conv_feats] #[(N,Co), ...]*len(Ks)
        cnn_feat = torch.cat(pooled_feats, 1) # (batch_size, feat_num)
        
        if self.dropout_p:
            cnn_feat = self.dropout(cnn_feat)
        #return cnn_feat, None, cnn_feat
        outputs = self.tanh(self.output_layer(cnn_feat))

        return (outputs, embeded.squeeze(1))
