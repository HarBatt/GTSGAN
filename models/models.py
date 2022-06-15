import torch
import torch.nn as nn

class Embedder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(Embedder, self).__init__()
        self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, hidden_size)
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
        self.rnn = nn.GRU(hidden_size, hidden_size, num_layers, batch_first=True)
        self.block_0 = self.mlp(hidden_size, input_size, False)
        self.block_1 = self.mlp(input_size, input_size, True)

        nn.init.xavier_uniform_(self.block_0[0].weight)
        nn.init.xavier_uniform_(self.block_1[0].weight)

    
    def mlp(self, input_size, hidden_size, final_layer):
        layers = nn.ModuleList([
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
        ])
        if final_layer:
            layers.append(nn.Sigmoid())
        else:
            layers.append(nn.LeakyReLU(0.2))
        return layers

    def forward(self, x):
        x, hidden = self.rnn(x)
        for layer in self.block_0:
            x = layer(x)
        for layer in self.block_1:
            x = layer(x)
        return x

class Generator(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(Generator, self).__init__()
        self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, hidden_size)
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
        self.rnn = nn.GRU(hidden_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, hidden_size)
        self.sigmoid = nn.Sigmoid()
        nn.init.xavier_uniform_(self.linear.weight)
        
    def forward(self, x):
        x, hidden = self.rnn(x)
        x = self.linear(x)
        x = self.sigmoid(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, hidden_size, num_layers):
        super(Discriminator, self).__init__()
        self.rnn = nn.GRU(hidden_size, hidden_size, num_layers, bidirectional = True, batch_first=True)
        self.block_0 = self.mlp(2*hidden_size, hidden_size, False)
        self.block_1 = nn.Linear(hidden_size, 1)

        nn.init.xavier_uniform_(self.block_0[0].weight)
        nn.init.xavier_uniform_(self.block_1.weight)

    
    def mlp(self, input_size, hidden_size, final_layer):
        layers = nn.ModuleList([
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
        ])
        if final_layer:
            layers.append(nn.Sigmoid())
        else:
            layers.append(nn.LeakyReLU(0.2))
        return layers

    def forward(self, x):
        x, hidden = self.rnn(x)
        for layer in self.block_0:
            x = layer(x)
        x = self.block_1(x)
        return x


class FusionBlock(nn.Module):
    def __init__(self, params):
        super(FusionBlock, self).__init__()
        self.params = params
        self.block = self.mlp(params.num_nodes + params.hidden_size, params.hidden_size)
        self.feature_pool = nn.Conv1d(params.seq_len, 1, kernel_size = 1)
    
    def mlp(self, input_size, hidden_size):
        layers = nn.ModuleList([
            nn.Linear(input_size, hidden_size),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid()
        ])
        return layers
    
    def structure_pool(self, embedding):
        embedding = embedding.permute(0, 2, 1)
        embedding = self.feature_pool(embedding)
        embedding = embedding.repeat(1, self.params.seq_len, 1)
        
        return embedding


    def forward(self, feature_embedding, structural_embedding):
        structural_embedding = self.structure_pool(structural_embedding)
        fusion_embedd = torch.cat([feature_embedding, structural_embedding], dim = 2)
        for layer in self.block:
                fusion_embedd = layer(fusion_embedd)

        return fusion_embedd


