import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp

from HGTAN.layers import Attn_head, Attn_head_adj



class hyperedge_attn(nn.Module):
    def __init__(self, nfeat, nhid, dropout):
        """ Intra-hyperedge attention module. """

        super(hyperedge_attn, self).__init__()


        self.intra_hpyeredge = Attn_head(nfeat, nhid, in_drop=0.6, coef_drop=0.6, activation=nn.ELU(),
                                        residual=True)
        self.industry_tensor = Attn_head_adj(nfeat, nhid, in_drop=0.6, coef_drop=0.6, activation=nn.ELU(),
                                             residual=True)
        self.dropout = dropout

    def forward(self, x, H, adj, nhid):
        batch = x.size(0)
        stock = x.size(1)
        industry_tensor = self.industry_tensor(x, adj)

        all_hyperedge_fts = torch.randn(0).cuda()
        hyperedge_fts = torch.randn(0).cuda()

        for i in range(H.shape[1]):
            intra_hyperedge_fts = torch.randn(0).cuda()
            node_set = torch.nonzero(H[:, i], as_tuple=False)

            for j in range(len(node_set)):
                node_index = node_set[j]
                intra_hyperedge_fts = torch.cat([intra_hyperedge_fts, x[:, node_index, :]], dim=1)

            after_intra = self.intra_hpyeredge(intra_hyperedge_fts)
            pooling = torch.nn.MaxPool1d(len(node_set), stride=1)
            e_fts = pooling(after_intra.permute(0, 2, 1))
            hyperedge_fts = torch.cat([hyperedge_fts, e_fts.permute(0, 2, 1)], dim=1)

            single_edge = torch.zeros(batch, stock, nhid).cuda()

            for j in range(len(node_set)):
                node_index = node_set[j]
                single_edge[:, node_index.squeeze(0), :] = after_intra[:, j, :]

            all_hyperedge_fts = torch.cat([all_hyperedge_fts, single_edge.unsqueeze(0)], dim=0)

        return all_hyperedge_fts, hyperedge_fts, industry_tensor



