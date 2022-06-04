"""
(0) MinMaxScaler: Min Max normalizer
(2) real_data_loading: Load and preprocess real data
"""

## Necessary Packages
import numpy as np


def MinMaxScaler(data):
  """Min Max normalizer.
  
  Args:
    - data: original data
  
  Returns:
    - norm_data: normalized data
  """
  minimum, maximum = np.min(data, 0), np.max(data, 0)
  numerator = data - minimum
  denominator = maximum - minimum
  norm_data = numerator / (denominator + 1e-7)
  return norm_data, (minimum, maximum)


  
def real_data_loading (absolute_path, seq_len):
  """Load and preprocess real-world datasets.
  
  Args:
    - absolute_path: absoulte_path
    - seq_len: sequence length
    
  Returns:
    - data: preprocessed data.
  """  

  ori_data = np.loadtxt(absolute_path, delimiter = ",",skiprows = 1)
  # Flip the data to make chronological data
  ori_data = ori_data[::-1]
  # Normalize the data
  ori_data, (minimum, maximum) = MinMaxScaler(ori_data)
    
  # Preprocess the dataset
  temp_data = []    
  # Cut data by sequence length
  for i in range(0, len(ori_data) - seq_len):
    _x = ori_data[i:i + seq_len]
    temp_data.append(_x)
        
  # Mix the datasets (to make it similar to i.i.d)
  idx = np.random.permutation(len(temp_data))    
  data = []
  for i in range(len(temp_data)):
    data.append(temp_data[idx[i]])
    
  return np.array(data), (minimum, maximum)