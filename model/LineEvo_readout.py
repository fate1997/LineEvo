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
    def __init__(self, dim=128, dropout=0, num_layers=1, if_pos=False):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(LineEvoLayer(dim, dropout, if_pos))
        
    
    def forward(self, x, edge_index, edge_attr, pos, batch):
        edges = torch.as_tensor(np.array(nx.from_edgelist(edge_index.T.tolist()).edges)).to(x.device)

        mol_repr_all = 0
        for layer in self.layers:
            x, edges, edge_attr, pos, batch, mol_repr = layer(x, edges, edge_attr, pos, batch)
            mol_repr_all = mol_repr_all + mol_repr
        
        return mol_repr_all


class LineEvoLayer(nn.Module):
    def __init__(self, dim=128, dropout=0.1, if_pos=False):
        super().__init__()
        self.dim = dim
        self.if_pos = if_pos

        # feature evolution
        self.linear = nn.Linear(dim, dim)
        # self.bias = nn.Parameter(torch.Tensor(dim))
        self.act = nn.ELU()
        self.dropout = nn.Dropout(dropout)
        self.attn = nn.Parameter(torch.randn(1, dim))

        if self.if_pos:
            self.rbf_expand = RBFExpansion(0, 5, 6)
            self.linear_rbf = nn.Linear(6, dim, bias=False)

        self.init_params()
        # readout phase
        self.readout = ReadoutPhase(dim)
    
    def init_params(self):
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.xavier_uniform_(self.attn)
        nn.init.zeros_(self.linear.bias)

        if self.if_pos:
            nn.init.xavier_uniform_(self.linear_rbf.weight)        


    def forward(self, x, edges, edge_attr, pos, batch):
        
        # create edges for isolated nodes
        num_nodes = x.shape[0]
        isolated_nodes = set(range(num_nodes)).difference(set(edges.flatten().tolist()))
        edges = torch.cat([edges, torch.LongTensor([[i, i] for i in isolated_nodes]).to(edges.device)], dim=0).to(torch.long)

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
        edges = self.edge_evolve(edges.to(x.device))
        
        # final readout
        mol_repr = self.readout(atom_repr, batch)
        
        return atom_repr, edges, edge_attr, pos, batch, mol_repr
    

    def edge_evolve(self, edges):
        l = edges[:, 0].tolist()+ edges[:, 1].tolist()
        tally = defaultdict(list)
        for i, item in enumerate(l):
            tally[item].append(i if i < len(l)//2 else i - len(l)//2)
        
        output = []
        for _, locs in tally.items():
            if len(locs) > 1:
                output.append(list(combinations(locs, 2)))
        
        return torch.tensor(list(chain(*output))).to(edges.device)