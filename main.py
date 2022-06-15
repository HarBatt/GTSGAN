## Necessary packages
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import warnings
warnings.filterwarnings("ignore")

# 1. GTSGAN model
from gtsgan import gtsgan
# 2. Data loading
from data_loading import real_data_loading
# 3. Utils
from utils import Parameters

#set the device 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

## Data loading
data_path = "data/"
dataset = "stock"
path_real_data = "data/" + dataset + "_data.csv"

#parameters

params = Parameters()
params.dataset = dataset
params.data_path = "data/" + params.dataset + "_data.csv"
params.model_save_path = "saved_models/" + params.dataset
params.seq_len = 24
params.batch_size = 128
params.max_steps = 10000
params.gamma = 1.0
params.save_model = True
params.print_every = 1000
params.device = "cuda"
params.save_synth_data = True


#preprocessing the data.

"""
Method: real_data_loading()
---------------------------------------------------------------------------------------------------------------------
    - Loads the data from the path.
    - Scales the data using a min-max scaler.
    - Slices the data into windows of size seq_len.
    - Shuffles the data randomly, and returns it.
"""
ori_data, (minimum, maximum) = real_data_loading(path_real_data, params.seq_len)

params.input_size = ori_data[0].shape[1]
params.hidden_size = 16
params.disc_out_size = 1
params.num_layers = 3


########################## Additional Parameters for Graph Encoder ##########################
params.num_nodes = params.input_size
params.graph_hidden = params.seq_len
params.graph_input = params.seq_len
params.top_k = params.input_size - 1
params.return_attention = True
params.get_graph = True

print('Preprocessing Complete!')
   
with open(data_path + params.dataset + '_real_data.npy', 'wb') as f:
    np.save(f, np.array(ori_data))

print("Saved real data! {}".format(params.dataset))

# Run GTSGAN
"""
Method: gtsgan()
---------------------------------------------------------------------------------------------------------------------
    - Runs the gtsgan model.
"""

generated_data, new_edge_index, learned_graph = gtsgan(ori_data, params) 
# print(new_edge_index.shape)
# print(learned_graph.shape) 

# # Renormalization
# generated_data = generated_data*maximum
# generated_data = generated_data + minimum 

if params.save_synth_data:
    with open(data_path + params.dataset + '_synthetic_data.npy', 'wb') as f:
        np.save(f, np.array(generated_data))

if params.return_attention:
    with open(data_path + params.dataset + '_attention_weights.npy', 'wb') as f:
        np.save(f, np.array(new_edge_index))

if params.get_graph:
    with open(data_path + params.dataset + '_graph.npy', 'wb') as f:
        np.save(f, np.array(learned_graph))