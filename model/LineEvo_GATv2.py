from os import read
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import global_add_pool
from torch_geometric.utils import to_undirected
from .modules import MLP, ReadoutPhase, RBFExpansion


class GLN(nn.Module):
    def __init__(self, config=None):
        super(GLN, self).__init__()

        self.input_dim = config.model.input_dim
        self.hidden_dim = config.model.hidden_dim
        self.num_heads = config.model.num_heads
        self.dropout = config.model.dropout
        self.output_dim = config.model.output_dim
        self.line_dropout = 0.1
        self.if_pos = config.model.readout.if_pos

        self.arrange = 'GGLGGL'

        self.layers = nn.ModuleList()
        for i, ind in enumerate(list(self.arrange)):
            if ind == 'G':
                layer = GATv2Layer(
                    num_node_features=self.input_dim if i == 0 else self.hidden_dim,
                    output_dim=self.hidden_dim // self.num_heads,
                    num_heads=self.num_heads,
                    concat=True,
                    activation=nn.PReLU(),
                    residual=True,
                    bias=True,
                    dropout=self.dropout)
                self.layers.append(layer)
            
            elif ind == 'L':
                layer = LineEvoLayer(
                    in_dim=self.input_dim if i == 0 else self.hidden_dim,
                    dim=self.hidden_dim, 
                    dropout=self.line_dropout, 
                    if_pos=self.if_pos)
                self.layers.append(layer)
            
            else:
                raise ValueError(f'Indicator should be G or L.')
        
        # prediction phase
        self.predict = MLP(self.hidden_dim*2, 128, self.output_dim, 2, 0.2, nn.ELU())
            
    def forward(self, data: Data):
        x, edge_index, edge_attr, pos, batch = data.x, data.edge_index, data.edge_attr, data.pos, data.batch
        
        mol_repr_all = 0
        line_index = 0
        for i, ind in enumerate(list(self.arrange)):
            if ind == 'G':
                x, mol_repr = self.layers[i](x, edge_index, batch)
            elif ind == 'L':
                edges = getattr(data, f'edges_{line_index}')
                x, pos, batch, mol_repr = self.layers[i](x, edges, pos, batch)
                line_index += 1
                evolve_edges = getattr(data, f'edges_{line_index}')
                edge_index = to_undirected(edges.T)
            mol_repr_all += mol_repr
        
        return self.predict(mol_repr_all)


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


class GATv2Layer(nn.Module):
    def __init__(self, num_node_features: int, output_dim: int, num_heads: int,
                 activation=nn.PReLU(), concat: bool = True, residual: bool = True,
                 bias: bool = True, dropout: float = 0.1, share_weights: bool = False):
        super(GATv2Layer, self).__init__()

        self.num_node_features = num_node_features
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.residual = residual
        self.activation = activation
        self.concat = concat
        self.dropout = dropout
        self.share_weights = share_weights

        # Embedding by linear projection
        self.linear_src = nn.Linear(num_node_features, output_dim * num_heads, bias=False)
        if self.share_weights:
            self.linear_dst = self.linear_src
        else:
            self.linear_dst = nn.Linear(num_node_features, output_dim * num_heads, bias=False)


        # The learnable parameters to compute attention coefficients
        self.double_attn = nn.Parameter(torch.Tensor(1, num_heads, output_dim))

        # Bias and concat
        if bias and concat:
            self.bias = nn.Parameter(torch.Tensor(output_dim * num_heads))
        elif bias and not concat:
            self.bias = nn.Parameter(torch.Tensor(output_dim))
        else:
            self.register_parameter('bias', None)

        if residual:
            if num_node_features == num_heads * output_dim:
                self.residual_linear = nn.Identity()
            else:
                self.residual_linear = nn.Linear(num_node_features, num_heads * output_dim, bias=False)
        else:
            self.register_parameter('residual_linear', None)

        # Some fixed function
        self.leakyReLU = nn.LeakyReLU(negative_slope=0.2)
        self.activation = activation
        self.dropout = nn.Dropout(dropout)

        # Readout
        self.readout = ReadoutPhase(output_dim * num_heads)

        self.init_params()

    def init_params(self):
        nn.init.xavier_uniform_(self.linear_src.weight)
        nn.init.xavier_uniform_(self.linear_dst.weight)

        nn.init.xavier_uniform_(self.double_attn)
        if self.residual:
            if self.num_node_features != self.num_heads * self.output_dim:
                nn.init.xavier_uniform_(self.residual_linear.weight)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)

    def forward(self, x, edge_index, batch):

        # Input preprocessing
        edge_src_index, edge_dst_index = edge_index

        # Projection on the new space
        src_projected = self.linear_src(self.dropout(x)).view(-1, self.num_heads, self.output_dim)
        dst_projected = self.linear_dst(self.dropout(x)).view(-1, self.num_heads, self.output_dim)
        

        #######################################
        ############## Edge Attn ##############
        #######################################

        # Edge attention coefficients
        edge_attn = self.leakyReLU((src_projected.index_select(0, edge_src_index)
                                    + dst_projected.index_select(0, edge_dst_index)))
        edge_attn = (self.double_attn * edge_attn).sum(-1)
        exp_edge_attn = (edge_attn - edge_attn.max()).exp()

        # sum the edge scores to destination node
        num_nodes = x.shape[0]
        edge_node_score_sum = torch.zeros([num_nodes, self.num_heads],
                                          dtype=exp_edge_attn.dtype,
                                          device=exp_edge_attn.device)
        edge_dst_index_broadcast = edge_dst_index.unsqueeze(-1).expand_as(exp_edge_attn)
        edge_node_score_sum.scatter_add_(0, edge_dst_index_broadcast, exp_edge_attn)

        # normalized edge attention
        # edge_attn shape = [num_edges, num_heads, 1]
        exp_edge_attn = exp_edge_attn / (edge_node_score_sum.index_select(0, edge_dst_index) + 1e-16)
        exp_edge_attn = self.dropout(exp_edge_attn).unsqueeze(-1)

        # summation from one-hop atom
        edge_x_projected = src_projected.index_select(0, edge_src_index) * exp_edge_attn
        edge_output = torch.zeros([num_nodes, self.num_heads, self.output_dim],
                                  dtype=exp_edge_attn.dtype,
                                  device=exp_edge_attn.device)
        edge_dst_index_broadcast = (edge_dst_index.unsqueeze(-1)).unsqueeze(-1).expand_as(edge_x_projected)
        edge_output.scatter_add_(0, edge_dst_index_broadcast, edge_x_projected)

        output = edge_output
        # residual, concat, bias, activation
        if self.residual:
            output += self.residual_linear(x).view(num_nodes, -1, self.output_dim)
        if self.concat:
            output = output.view(-1, self.num_heads * self.output_dim)
        else:
            output = output.mean(dim=1)

        if self.bias is not None:
            output += self.bias

        if self.activation is not None:
            output = self.activation(output)

        return output, self.readout(output, batch)