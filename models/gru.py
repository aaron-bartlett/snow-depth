import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


class GRUForecaster(nn.Module):


  def __init__(self, n_features, n_hidden, n_outputs, sequence_len, n_gru_layers=1, n_deep_layers=10, use_cuda=False, dropout=0.2):
    '''
    n_features: number of input features (1 for univariate forecasting)
    n_hidden: number of neurons in each hidden layer
    n_outputs: number of outputs to predict for each training example
    n_deep_layers: number of hidden dense layers after the gru layer
    sequence_len: number of steps to look back at for prediction
    dropout: float (0 < dropout < 1) dropout ratio between dense layers
    '''
    super().__init__()

    self.n_gru_layers = n_gru_layers
    self.nhid = n_hidden
    self.use_cuda = use_cuda
    self.device = "cuda" if use_cuda else "cpu"

    # GRU Layer
    self.gru = nn.GRU(n_features,
                        n_hidden,
                        num_layers=n_gru_layers,
                        batch_first=True)
    
    # first dense after gru
    self.fc1 = nn.Linear(n_hidden * sequence_len, n_hidden) 
    # Dropout layer 
    self.dropout = nn.Dropout(p=dropout)

    # Fully connected layers (n_hidden x n_deep_layers)
    dnn_layers = []
    for i in range(n_deep_layers):
      # Last layer (n_hidden x n_outputs)
      if i == n_deep_layers - 1:
        dnn_layers.append(nn.ReLU())
        dnn_layers.append(nn.Linear(self.nhid, n_outputs))
      # All other layers (n_hidden x n_hidden) with dropout option
      else:
        dnn_layers.append(nn.ReLU())
        dnn_layers.append(nn.Linear(self.nhid, self.nhid))
        if dropout:
          dnn_layers.append(nn.Dropout(p=dropout))
    # compile DNN layers
    self.dnn = nn.Sequential(*dnn_layers)

  def forward(self, x):

    # Init hidden state
    hidden_state = torch.zeros(self.n_gru_layers, x.shape[0], self.nhid)
    #cell_state = torch.zeros(self.n_gru_layers, x.shape[0], self.nhid)

    # move hidden state to device
    if self.use_cuda:
      hidden_state = hidden_state.to(self.device)
        
    self.hidden = hidden_state

    # Forward Pass
    x, _ = self.gru(x, self.hidden) 
    x = self.dropout(x.contiguous().view(x.shape[0], -1)) 
    x = self.fc1(x) 
    return self.dnn(x)