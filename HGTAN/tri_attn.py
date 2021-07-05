"""Hypergraph Tri-attention Network."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
from HGTAN.hyperedge_attn import hyperedge_attn
from HGTAN.layers import edge_attn


class HGAT(nn.Module):
    """Tri-attention Modules."""

    def __init__(self, nfeat, nhid, dropout):

        super(HGAT, self).__init__()

        self.hyperedge = hyperedge_attn(nfeat, nhid, dropout)
        self.attn = edge_attn(nhid, nhid*2)
        self.dropout = dropout

    def forward(self, x, H, adj,nhid):

        x = F.dropout(x, self.dropout, training=self.training)

        all_hyperedge_tensor, hyperedge_tensor, industry_tensor = self.hyperedge(x, H, adj, nhid)
        all_hyperedge_tensor = F.elu(all_hyperedge_tensor)
        hyperedge_tensor = F.elu(hyperedge_tensor)

        final_tensor = torch.randn(0).cuda()

        for i in range(x.shape[1]):
            final_fts_i = torch.randn(0).cuda()
            hyperedge_fts_i = torch.randn(0).cuda()

            if torch.sum(H, 1)[i] > 1:
                # vertex degree > 1
                hyperedge = torch.nonzero(H[i], as_tuple=False)

                for j in range(len(hyperedge)):
                    hyperedge_num = hyperedge[j]
                    final_fts_i = torch.cat([final_fts_i, all_hyperedge_tensor[hyperedge_num, :, i, :].permute(1, 0, 2)], dim=1)
                    hyperedge_fts_i = torch.cat([hyperedge_fts_i, hyperedge_tensor[:, hyperedge_num, :]], dim=1)

                coefs = self.attn(hyperedge_fts_i)

                final_fts_i = torch.matmul(coefs, final_fts_i)
                indus_fund = torch.cat([industry_tensor[:, i, :].unsqueeze(1), final_fts_i], dim=1)

                coefs = self.attn(indus_fund)
                final_indus_fund = torch.matmul(coefs, indus_fund)
                final_tensor = torch.cat([final_tensor, final_indus_fund], dim=1)

            else:
                hyperedge = torch.nonzero(H[i], as_tuple=False)
                hyperedge_num = (hyperedge.squeeze(0)).squeeze(0)
                indus_fund = torch.cat([industry_tensor[:, i, :].unsqueeze(1),
                                        all_hyperedge_tensor[hyperedge_num, :, i, :].unsqueeze(1)], dim=1)
                coefs = self.attn(indus_fund)
                final_indus_fund = torch.matmul(coefs, indus_fund)
                final_tensor = torch.cat([final_tensor, final_indus_fund], dim=1)

        x = F.dropout(final_tensor, self.dropout, training=self.training)

        return x


