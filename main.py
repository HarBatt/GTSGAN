## Necessary packages
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import warnings
warnings.filterwarnings("ignore")

# 1. RGAN model
from timegan import timegan
# 2. Data loading
from data_loading import real_data_loading
# 3. Metrics
from metrics.visualization_metrics import visualization
# 4. Utils
from utils import Parameters




#set the device 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

## Data loading
data_path = "data/"
dataset = "energy"
path_real_data = "data/" + dataset + "_data.csv"
#Evaluation of the model, by default can be set to false.
eval_model = False

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
params.print_every = 500
params.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
params.save_synth_data = False

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
params.hidden_size = 24
params.num_layers = 3

print('Preprocessing Complete!')
   
with open(data_path + params.dataset + '_real_data.npy', 'wb') as f:
    np.save(f, np.array(ori_data))

print("Saved real data!")

# Run TimeGAN
"""
Method: timegan()
---------------------------------------------------------------------------------------------------------------------
    - Runs the timegan model.
"""
generated_data = timegan(ori_data, params)  

# # Renormalization
# generated_data = generated_data*maximum
# generated_data = generated_data + minimum 

with open(data_path + params.dataset + '_synthetic_data.npy', 'wb') as f:
    np.save(f, np.array(generated_data))