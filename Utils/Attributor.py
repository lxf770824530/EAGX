import torch
import torch.nn as nn
from torch.nn import ReLU, Sequential
from torch_geometric.nn import Linear, LayerNorm
import torch.nn.functional as F

import os
from config import arg_parse
args = arg_parse()

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]='0'
device = args.device



class Attributor(nn.Module):

    def __init__(self, _input_dim, _centers, _sigma):
        super(Attributor, self).__init__()

        self.input_dim = _input_dim

        self.LN = LayerNorm(1)
        self.relu = ReLU()
        self.linear1 = Linear(self.input_dim, 32)
        self.linear2 = Linear(32, 1)

        self.centers = _centers
        self.sigma = _sigma


    def get_centers(self):
        return self.centers

    def get_sigmas(self):
        return self.sigma

    def forward(self, edge_emb, counter_edge_graph_emb):

        x = counter_edge_graph_emb.view(counter_edge_graph_emb.size()[0], 1,-1).expand(-1, self.centers.size()[0], -1)
        mu = torch.exp(- (x - self.centers) ** 2 / (2 * self.sigma ** 2))
        x = mu.view(mu.size()[0],-1)
        x = torch.cat([edge_emb, x], dim=1)


        x = self.LN(x)
        x = self.relu(x)
        x = self.linear1(x)
        x = self.LN(x)
        x = self.relu(x)
        x = self.linear2(x)

        return x









