from sympy import DiagMatrix
import torch
from torch import nn
from torch_geometric.nn import global_add_pool

import numpy as np
import networkx as nx

from collections import defaultdict
from itertools import combinations, chain

from .modules import RBFExpansion, ReadoutPhase



class LineEvo(nn.Module):
    def __init__(self, in_dim=63, dim=128, dropout=0, num_layers=1, if_pos=True):
        super().__init__()
        self.dim = dim
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(LineEvoLayer(in_dim if i==0 else dim, dim, dropout, if_pos))
        
    
    def forward(self, data):
        x, pos, batch = data.x, data.pos, data.batch

        mol_repr_all = 0
        for i, layer in enumerate(self.layers):
            edges = getattr(data, f'edges_{i}')
            x, pos, batch, mol_repr = layer(x, edges, pos, batch)
            mol_repr_all = mol_repr_all + mol_repr
        
        return mol_repr_all # 在算SchNet的时候这个好像是mol_repr


class LineEvoLayer(nn.Module):
    def __init__(self, in_dim=128, dim=128, dropout=0.1, if_pos=False):
        super().__init__()
        self.dim = dim
        self.if_pos = if_pos

        # feature evolution
        self.linear = nn.Linear(in_dim, dim)
        # self.bias = nn.Parameter(torch.Tensor(dim))
        self.act = nn.ELU()
        self.dropout = nn.Dropout(dropout)
        self.attn = nn.Parameter(torch.randn(1, dim))

        if self.if_pos:
            num_gaussians = 6
            self.rbf_expand = RBFExpansion(0, 5, num_gaussians)
            self.linear_rbf = nn.Linear(num_gaussians, dim, bias=False)

        self.init_params()
        # readout phase
        self.readout = ReadoutPhase(dim)
    
    def init_params(self):
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.xavier_uniform_(self.attn)
        nn.init.zeros_(self.linear.bias)

        if self.if_pos:
            nn.init.xavier_uniform_(self.linear_rbf.weight)        


    def forward(self, x, edges, pos, batch):
        
        # feature evolution
        x = self.dropout(x)
        x_src = self.linear(x).index_select(0, edges[:, 0])
        x_dst = self.linear(x).index_select(0, edges[:, 1])
        x = self.act(x_src + x_dst)
        
        if self.if_pos:
            pos_src = pos.index_select(0, edges[:, 0])
            pos_dst = pos.index_select(0, edges[:, 1])
            vector = pos_dst - pos_src
            distance = torch.norm(vector, p=2, dim=1).unsqueeze(-1)
            torch.clamp_(distance, min=0.1)
            distance_matrix = self.rbf_expand(distance)
            dist_emd = self.linear_rbf(distance_matrix)
            x = x * dist_emd
            pos = (pos_src + pos_dst) / 2
        atom_repr = x * self.attn
        
        # test
        atom_repr = nn.ELU()(atom_repr)

        # update batch and edges
        batch = batch.index_select(0, edges[:, 0])
        
        # final readout
        mol_repr = self.readout(atom_repr, batch)
        
        return atom_repr, pos, batch, mol_repr
