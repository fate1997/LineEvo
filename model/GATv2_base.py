from os import read
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import global_add_pool

from .modules import MLP, ReadoutPhase
from .LineEvo_GATv2 import GATv2Layer
from .LineEvo_readout import LineEvo


class GATv2(nn.Module):
    def __init__(self, config=None):
        super(GATv2, self).__init__()

        self.input_dim = config.model.input_dim
        self.hidden_dim = config.model.hidden_dim
        self.num_layers = config.model.num_layers
        self.num_heads = config.model.num_heads
        self.dropout = config.model.dropout
        self.output_dim = config.model.output_dim
        self.readout = config.model.readout

        # GAT layers
        self.layers = nn.ModuleList()
        for i in range(self.num_layers):
            layer = GATv2Layer(
                num_node_features=self.input_dim if i == 0 else self.hidden_dim,
                output_dim=self.hidden_dim // self.num_heads,
                num_heads=self.num_heads,
                concat=True,
                activation=nn.PReLU(),
                residual=True,
                bias=True,
                dropout=self.dropout
            )
            self.layers.append(layer)

        # Readout phase
        self.readout_func = self.get_readout(self.readout)

        # prediction phase
        self.predict = MLP(2*self.hidden_dim, 128, self.output_dim, 2, 0.2, nn.ELU())


    def forward(self, data: Data):
        x, edge_index, edge_attr, pos, batch = data.x, data.edge_index, data.edge_attr, data.pos, data.batch
        
        mol_repr_all = 0
        for i, layer in enumerate(self.layers):
            x, mol_repr = layer(x, edge_index, batch)
            mol_repr_all += mol_repr

        if self.readout.name == 'LineEvo':
            mol_repr = self.readout_func(x, edge_index, edge_attr, pos, batch)
            mol_repr_all += mol_repr

        return self.predict(mol_repr_all) # mol_repr_all
    

    def get_readout(self, readout_config):
        readout_name = readout_config.name

        if readout_name == 'LineEvo':
            return LineEvo(self.hidden_dim, readout_config.dropout, readout_config.num_layers, readout_config.if_pos)
        elif readout_name == 'Add':
            return ReadoutPhase(self.hidden_dim)
        else:
            pass

    def load_config(self, config):
        for key in config.model.keys():
            if key in self.__dict__:
                self.__dict__[key] = config.model[key]