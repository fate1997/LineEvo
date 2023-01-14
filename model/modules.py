from torch import nn
import torch
import numpy as np
from torch.autograd import Variable
from torch_geometric.nn import global_add_pool, global_max_pool, global_mean_pool

from typing import Tuple

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, layers, dropout, activation):
        super().__init__()
        if layers == 1:
            self.layers = nn.Linear(input_dim, output_dim)
        else:
            self.layers = []
            for i in range(layers - 1):
                self.layers.append(nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim))
                self.layers.append(activation)
                self.layers.append(nn.LayerNorm(hidden_dim))
                self.layers.append(nn.Dropout(dropout))

            self.layers.append(nn.Linear(hidden_dim, output_dim))
            self.layers = nn.Sequential(*self.layers)
        
        self.layers.apply(init_weight)
        
    def forward(self, x):
        output = self.layers(x)
        return output


def init_weight(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias != None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


class RBFExpansion(nn.Module):
    def __init__(self, start=0.0, stop=5.0, num_gaussians=50):
        super(RBFExpansion, self).__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item()**2
        self.register_buffer('offset', offset)

    def forward(self, dist):
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


        
class PositionEncoder(nn.Module):
    def __init__(self, d_model, seq_len=4, device='cuda:0'):
        super().__init__()
        # position_enc.shape = [seq_len, d_model]
        position_enc = np.array([[pos / np.power(10000, 2 * (j // 2) / d_model) for j in range(d_model)] for pos in range(seq_len)])
        position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])
        position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])
        self.position_enc = torch.tensor(position_enc, device=device).unsqueeze(0).float()

    def forward(self, x):
        # x.shape = [batch_size, seq_length, d_model]
        x = x * Variable(self.position_enc, requires_grad=False)
        return x


class ReadoutPhase(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # readout phase
        self.weighting = nn.Linear(dim, 1) 
        self.score = nn.Sigmoid() 
        
        nn.init.xavier_uniform_(self.weighting.weight)
        nn.init.constant_(self.weighting.bias, 0)
    
    def forward(self, x, batch):
        weighted = self.weighting(x)
        score = self.score(weighted)
        output1 = global_add_pool(score * x, batch)
        output2 = global_max_pool(x, batch)
        
        output = torch.cat([output1, output2], dim=1)
        return output