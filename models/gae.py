import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv


class GraphEncoder(nn.Module):
    def __init__(self, params, input_size, hidden_size):
        super(GraphEncoder, self).__init__()
        self.params = params
        self.edge_index = self.dense_graph(params.input_size).to(params.device)
        self.graph_block = self.graph_sage_block(input_size, hidden_size)
        self.block_1 = self.linear_block(hidden_size, input_size)
        self.block_2 = self.linear_block(params.input_size, hidden_size)

        torch.nn.init.xavier_uniform_(self.block_1[0].weight)
        torch.nn.init.xavier_uniform_(self.block_1[1].weight)
        torch.nn.init.xavier_uniform_(self.block_2[0].weight)
        torch.nn.init.xavier_uniform_(self.block_2[1].weight)
    
    def graph_sage_block(self, input_size, hidden_size):
        layers = nn.ModuleList([
            SAGEConv(input_size, hidden_size), 
            SAGEConv(hidden_size, hidden_size),
        ])
        return layers
    
    def linear_block(self, hidden_size, input_size):
        layers = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size, bias = False),
            nn.Linear(hidden_size, input_size, bias = False),
            nn.Sigmoid(),
        ])
        return layers
    
    def dense_graph(self, num_nodes):
        adj_matrix = []
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i!=j:
                    adj_matrix.append([i, j])
        
        return torch.tensor(adj_matrix, dtype=torch.long).t().contiguous()
        
    def forward(self, x):
        x = x.permute(0, 2, 1)
        for layer in self.graph_block:
            x = layer(x, self.edge_index)
        for layer in self.block_1:
            x = layer(x)
        x = x.permute(0, 2, 1)

        for layer in self.block_2:
            x = layer(x)
        return x



