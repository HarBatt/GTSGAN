import torch.nn as nn
from models.graph_layers.gat_layer import GATLayer

class GNNLayer(nn.Module):
    def __init__(self, in_channel, out_channel, inter_dim=0, heads=1):
        super(GNNLayer, self).__init__()
        self.gnn = GATLayer(in_channel, out_channel, inter_dim=inter_dim, heads=heads, concat=False)
        self.linear = nn.Linear(out_channel, out_channel)
        self.bnorm = nn.BatchNorm1d(out_channel)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.att_weight = None 
        self.new_edge_index = None

    def forward(self, x, edge_index, embedding=None, node_num=0):
        out, (new_edge_index, att_weight) = self.gnn(x, edge_index, embedding, return_attention_weights=True)
        self.att_weight = att_weight
        self.new_edge_index = new_edge_index
        
        out = self.leaky_relu(out)
        out = self.linear(out)
        out = self.bnorm(out)
        return self.leaky_relu(out), self.att_weight, self.new_edge_index