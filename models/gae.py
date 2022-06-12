import math
import torch
import torch.nn as nn
from models.graph_layers.gnn_layer import GNNLayer

class GraphEncoder(nn.Module):
    def __init__(self, params, node_num, dim, input_dim, topk):
        super(GraphEncoder, self).__init__()
        embed_dim = dim
        self.node_embedding = None
        self.topk = topk
        self.learned_graph = None
        self.edge_index = self.dense_graph(node_num)
        self.embedding = nn.Embedding(node_num, embed_dim)
        self.bn_outlayer_in = nn.BatchNorm1d(embed_dim)
        self.gnn_layer = GNNLayer(input_dim, dim, inter_dim=dim+embed_dim, heads=1)
        self.cache_embed_index = None
        self.dp = nn.Dropout(0.2)

        # linear layers to transform the output of the graph encoder to hidden dim to make it consistent with the embedder's output
        self.linear_layers = self.linear_block(params.input_size, params.hidden_size)

        nn.init.kaiming_uniform_(self.embedding.weight, a=math.sqrt(5))
        nn.init.xavier_uniform_(self.linear_layers[0].weight, gain=1)
        nn.init.xavier_uniform_(self.linear_layers[1].weight, gain=1)

    def linear_block(self, input_size, hidden_size):
        layers = nn.ModuleList([
            nn.Linear(input_size, hidden_size, bias = False),
            nn.Linear(hidden_size, hidden_size, bias = True),
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


    def get_batch_edge_index(self, org_edge_index, batch_num, node_num):
        # org_edge_index:(2, edge_num)
        edge_index = org_edge_index.clone().detach()
        edge_num = org_edge_index.shape[1]
        batch_edge_index = edge_index.repeat(1,batch_num).contiguous()

        for i in range(batch_num):
            batch_edge_index[:, i*edge_num:(i+1)*edge_num] += i*node_num

        return batch_edge_index.long()
        
    def forward(self, data):
        device = data.device
        x = data.clone().detach()
        x = x.permute(0, 2, 1)
        batch_num, node_num, all_feature = x.shape
        x = x.reshape(-1, all_feature).contiguous()
        gcn_outs = []
    
        all_embeddings = self.embedding(torch.arange(node_num).to(device))
        weights_arr = all_embeddings.detach().clone()
        all_embeddings = all_embeddings.repeat(batch_num, 1)
        
        weights = weights_arr.view(node_num, -1)

        cos_ji_mat = torch.matmul(weights, weights.T)
        normed_mat = torch.matmul(weights.norm(dim=-1).view(-1,1), weights.norm(dim=-1).view(1,-1))
        cos_ji_mat = cos_ji_mat / normed_mat
        
        topk_num = self.topk
        topk_indices_ji = torch.topk(cos_ji_mat, topk_num, dim=-1)[1]

        self.learned_graph = topk_indices_ji

        gated_i = torch.arange(0, node_num).T.unsqueeze(1).repeat(1, topk_num).flatten().to(device).unsqueeze(0)
        gated_j = topk_indices_ji.flatten().unsqueeze(0)
        gated_edge_index = torch.cat((gated_j, gated_i), dim=0)
        batch_gated_edge_index = self.get_batch_edge_index(gated_edge_index, batch_num, node_num).to(device)
        gcn_out = self.gnn_layer(x, batch_gated_edge_index, node_num=node_num*batch_num, embedding=all_embeddings)
        gcn_outs.append(gcn_out)

        x = torch.cat(gcn_outs, dim=1)
        x = x.view(batch_num, node_num, -1)
        
        indexes = torch.arange(0,node_num).to(device)

        out = torch.mul(x, self.embedding(indexes))
        out = out.permute(0, 2, 1)

        for layer in self.linear_layers:
            out = layer(out)
        return out, self.embedding, self.learned_graph

