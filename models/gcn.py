import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math
import pdb

class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True, batch=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.batch = batch

    
    def initialize_weights(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        if self.batch:
            support = torch.matmul(input, self.weight)
            output = torch.matmul(adj, support)
        else:
            support = torch.mm(input, self.weight)
            output = torch.mm(adj, support)
            #output = SparseMM(adj)(support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN(nn.Module):
    def __init__(self, nin, nhid, nout, dropout=0.1):
        super(GCN, self).__init__()

        # self.gc_intra1 = GraphConvolution(nin, nhid, batch=True)
        # self.gc_intra2 = GraphConvolution(nhid, nout, batch=True)

        self.gc_inter1 = GraphConvolution(nin, nhid, batch=True)
        self.gc_inter2 = GraphConvolution(nhid, nout, batch=True)

        self.dropout = dropout

        # self.gc_intra1.initialize_weights()
        # self.gc_intra2.initialize_weights()
        self.gc_inter1.initialize_weights()
        self.gc_inter2.initialize_weights()

    def forward(self, x, adj_inter):

        # x = F.relu(self.gc_intra1(x, adj_intra))
        # x = F.relu(self.gc_inter1(x, adj_inter))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc_inter1(x, adj_inter))
        x = F.relu(self.gc_inter2(x, adj_inter))
        # x = F.relu(self.gc2(x, adj))
        # x = F.dropout(x, self.dropout, training=self.training)
        return x
