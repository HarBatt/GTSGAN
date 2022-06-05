import torch
import torch.nn as nn
from models.gae import GraphEncoder

class Embedder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(Embedder, self).__init__()
        self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, bias = False)
        self.linear = nn.Linear(hidden_size, hidden_size, bias = False)
        self.sigmoid = nn.Sigmoid()
        nn.init.xavier_uniform_(self.linear.weight)
        
    def forward(self, x):
        x, hidden = self.rnn(x)
        x = self.linear(x)
        x = self.sigmoid(x)
        return x

class Recovery(nn.Module):
    def __init__(self, hidden_size, input_size, num_layers):
        super(Recovery, self).__init__()
        self.rnn = nn.GRU(hidden_size, input_size, num_layers, batch_first=True, bias = False)
        self.linear = nn.Linear(input_size, input_size, bias = False)
        self.sigmoid = nn.Sigmoid()
        nn.init.xavier_uniform_(self.linear.weight)
        
    def forward(self, x):
        x, hidden = self.rnn(x)
        x = self.linear(x)
        x = self.sigmoid(x)
        return x

class Generator(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(Generator, self).__init__()
        self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, bias = False)
        self.linear = nn.Linear(hidden_size, hidden_size, bias = False)
        self.sigmoid = nn.Sigmoid()
        nn.init.xavier_uniform_(self.linear.weight)
        
    def forward(self, x):
        x, hidden = self.rnn(x)
        x = self.linear(x)
        x = self.sigmoid(x)
        return x

class Supervisor(nn.Module):
    def __init__(self, hidden_size, num_layers):
        super(Supervisor, self).__init__()
        self.rnn = nn.GRU(hidden_size, hidden_size, num_layers, batch_first=True, bias = False)
        self.linear = nn.Linear(hidden_size, hidden_size, bias = False)
        self.sigmoid = nn.Sigmoid()
        nn.init.xavier_uniform_(self.linear.weight)
        
    def forward(self, x):
        x, hidden = self.rnn(x)
        x = self.linear(x)
        x = self.sigmoid(x)
        return x

class Discriminator(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers):
        super(Discriminator, self).__init__()
        self.rnn = nn.GRU(hidden_size, hidden_size, num_layers, batch_first=True, bias = False, bidirectional = True)
        self.linear = nn.Linear(2*hidden_size, 1, bias = False)
        nn.init.xavier_uniform_(self.linear.weight)
        
    def forward(self, x):
        x, hidden = self.rnn(x)
        x = self.linear(x)
        return x

class FusionBlock(nn.Module):
    def __init__(self, temporal_embedd_size, structural_embedd_size):
        super(FusionBlock, self).__init__()
        self.linear = self.linear_block(temporal_embedd_size + structural_embedd_size, temporal_embedd_size)
        nn.init.xavier_uniform_(self.linear[0].weight)
        nn.init.xavier_uniform_(self.linear[1].weight)
    
    def linear_block(self, input_size, hidden_size):
        layers = nn.ModuleList([
            nn.Linear(input_size, hidden_size, bias = False),
            nn.Linear(hidden_size, hidden_size, bias = False),
            nn.Sigmoid()
        ])
        return layers

    def forward(self, temporal_embedding, structural_embedding):
        x = torch.cat([temporal_embedding, structural_embedding], dim = 2)
        for layer in self.linear:
            x = layer(x)
        return x


