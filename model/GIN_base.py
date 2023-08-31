from os import read
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import GINConv

from .modules import MLP, ReadoutPhase
from .LineEvo_readout import LineEvo


class GIN(nn.Module):
    def __init__(self, config=None):
        super(GIN, self).__init__()

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
            layer = GINConv(
                nn=MLP(self.input_dim if i==0 else self.hidden_dim, self.hidden_dim, self.hidden_dim, 2, 0.1, nn.ELU())
            )
            self.layers.append(layer)
        # Readout phase
        self.gin_readout = ReadoutPhase(self.hidden_dim)
        self.readout_func = self.get_readout(self.readout)

        # prediction phase
        self.predict = MLP(2*self.hidden_dim, 128, self.output_dim, 2, 0.2, nn.ELU())


    def forward(self, data: Data):
        x, edge_index, edge_attr, pos, batch = data.x, data.edge_index, data.edge_attr, data.pos, data.batch
        
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index)

        mol_repr_all = self.gin_readout(x, batch)
        
        if self.readout.name == 'LineEvo':
            data.x = data
            mol_repr = self.readout_func(data)
            mol_repr_all += mol_repr

        return self.predict(mol_repr_all) # mol_repr
    

    def get_readout(self, readout_config):
        readout_name = readout_config.name

        if readout_name == 'LineEvo':
            return LineEvo(self.hidden_dim, self.hidden_dim, readout_config.dropout, readout_config.num_layers, readout_config.if_pos)
        elif readout_name == 'Add':
            return ReadoutPhase(self.hidden_dim)
        else:
            pass


    def load_config(self, config):
        for key in config.model.keys():
            if key in self.__dict__:
                self.__dict__[key] = config.model[key]
