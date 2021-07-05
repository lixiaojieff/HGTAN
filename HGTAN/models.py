''' Define the HGTAN model '''
import torch

import torch.nn as nn
from HGTAN.tri_attn import HGAT
from HGTAN.temp_layers import TemporalAttention


def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''

    sz_b, len_s = seq.size(0), seq.size(1)
    subsequent_mask = torch.triu(
        torch.ones((len_s, len_s), device=seq.device, dtype=torch.uint8), diagonal=1)
    subsequent_mask = subsequent_mask.bool().bool().unsqueeze(0).expand(sz_b, -1, -1)

    return subsequent_mask




class HGTAN(nn.Module):
    ''' A sequence to sequence model with attention mechanism. '''

    def __init__(
            self,
            rnn_unit, n_hid, n_class,
            feature,
            d_word_vec, d_model,
            n_head, d_k, d_v, dropout,
            tgt_emb_prj_weight_sharing):

        super().__init__()

        self.linear = nn.Linear(feature, d_word_vec)
        self.rnn = nn.GRU(d_word_vec,
                          rnn_unit,
                          num_layers=2,
                          batch_first=True,
                          bidirectional=False)

        self.temp_attn = TemporalAttention(n_head, rnn_unit, d_k, d_v, dropout=dropout)

        self.HGAT = HGAT(nfeat=rnn_unit,
                         nhid=n_hid,
                         dropout=dropout,
                         )

        self.tgt_word_prj = nn.Linear(n_hid, n_class, bias=False)
        nn.init.xavier_normal_(self.tgt_word_prj.weight)

        assert d_model == d_word_vec, \
        'To facilitate the residual connections, \
         the dimensions of all module outputs shall be the same.'

        if tgt_emb_prj_weight_sharing:
            # Share the weight matrix between target word embedding & the final logit dense layer
            self.x_logit_scale = (d_model ** -0.5)
        else:
            self.x_logit_scale = 1.

    def forward(self, src_seq, H, adj, n_hid):
        src_seq = self.linear(src_seq)
        batch = src_seq.size(0)
        stock = src_seq.size(1)
        seq_len = src_seq.size(2)
        dim = src_seq.size(3)
        src_seq = torch.reshape(src_seq, (batch*stock, seq_len, dim))

        slf_attn_mask = get_subsequent_mask(src_seq).bool()

        rnn_output, *_ = self.rnn(src_seq)
        enc_output, enc_slf_attn = self.temp_attn(
            rnn_output, rnn_output, rnn_output, mask=slf_attn_mask.bool())

        enc_output = torch.reshape(enc_output, (batch, stock, -1))

        HGAT_output = self.HGAT(enc_output, H, adj, n_hid)


        HGAT_output = torch.reshape(HGAT_output, (batch*stock, -1))
        seq_logit = self.tgt_word_prj(HGAT_output) * self.x_logit_scale


        return seq_logit
