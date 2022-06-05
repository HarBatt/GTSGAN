## Necessary Packages
import numpy as np
import torch

""" utils.py

class: 
    - Parameters: store all the parameters

(1) train_test_divide: Divide train and test data for both original and synthetic data.
(2) extract_time: Returns Maximum sequence length and each sequence length.
(4) random_generator: random vector generator
(5) batch_generator: mini-batch generator
"""

class Parameters(object):
  def __init__(self):
    pass


def train_test_divide (data_x, data_x_hat, data_t, data_t_hat, train_rate = 0.8):
  """Divide train and test data for both original and synthetic data.
  
  Args:
    - data_x: original data
    - data_x_hat: generated data
    - data_t: original time
    - data_t_hat: generated time
    - train_rate: ratio of training data from the original data
  """
  # Divide train/test index (original data)
  no = len(data_x)
  idx = np.random.permutation(no)
  train_idx = idx[:int(no*train_rate)]
  test_idx = idx[int(no*train_rate):]
    
  train_x = [data_x[i] for i in train_idx]
  test_x = [data_x[i] for i in test_idx]
  train_t = [data_t[i] for i in train_idx]
  test_t = [data_t[i] for i in test_idx]      
    
  # Divide train/test index (synthetic data)
  no = len(data_x_hat)
  idx = np.random.permutation(no)
  train_idx = idx[:int(no*train_rate)]
  test_idx = idx[int(no*train_rate):]
  
  train_x_hat = [data_x_hat[i] for i in train_idx]
  test_x_hat = [data_x_hat[i] for i in test_idx]
  train_t_hat = [data_t_hat[i] for i in train_idx]
  test_t_hat = [data_t_hat[i] for i in test_idx]
  
  return train_x, train_x_hat, test_x, test_x_hat, train_t, train_t_hat, test_t, test_t_hat


def extract_time(data):
  """Returns Maximum sequence length and each sequence length.
  
  Args:
    - data: original data
    
  Returns:
    - time: extracted time information
    - max_seq_len: maximum sequence length
  """
  time = list()
  max_seq_len = 0
  for i in range(len(data)):
    max_seq_len = max(max_seq_len, len(data[i][:,0]))
    time.append(len(data[i][:,0]))
    
  return time, max_seq_len


def batch_generator(data, parameters):
  """Mini-batch generator.
  
  Args:
    - data: time-series data
    - parameters: the dictionary which stores all the parameters
    
  Returns:
    - X_mb: time-series data in each batch
  """
  batch_size = parameters.batch_size
  device = parameters.device
  def generate():
    #slicing of the 1st index(time)
    #ori_data = np.asarray(data)[:, :, 1:]
    #Some times you dont need to when there is no time index, ex: stock_data. 
    ori_data = np.asarray(data)[:, :, :]
    idx = np.random.permutation(len(ori_data)) 

    for i in generator_helper(idx, batch_size):
      X_mb = ori_data[i]
      X_mb = torch.from_numpy(X_mb).float().to(device)
      yield X_mb

  while True:
    for sample in generate():
      yield sample

def generator_helper(train_idx, batch_size):
    for i in range(0, len(train_idx), batch_size):
        data = train_idx[i:i + batch_size]
        if len(data) != batch_size:
            break
        yield data 