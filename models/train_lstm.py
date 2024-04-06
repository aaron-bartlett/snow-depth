import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, RobustScaler

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
torch.manual_seed(0)

import torchvision

from lstm import LSTMForecaster
from utils import *

if('-h' in sys.argv):
    print('Usage: python train_lstm.py <model num> [-n,-h]')
    sys.exit()

# Device selection (CPU | GPU)
USE_CUDA = torch.cuda.is_available()
device = 'metal' if USE_CUDA else 'cpu'
print(f"Using '{device}' device")


df = pd.read_csv("../data/large_dataset.csv", index_col="DATE")
df.index = pd.to_datetime(df.index)
#features_list = ["AWND","PRCP","SNOW","SNWD","TMAX","TMIN","WDFX","WSFX"]
features_list = ["AWND","PRCP","SNOW","SNWD","TMAX","TMIN","WDF2","WSF2"]
targets_list = ["SNWD"]
#df["WSFX"] = df[['WSF1', 'WSF2']].max(axis=1)
#df["WDFX"] = df[['WDF1', 'WDF2']].max(axis=1)
df = df[features_list].copy()
df = df.fillna(0)

USE_NORM = False
if('-n' in sys.argv):
    USE_NORM = True
    print('Using Normalized Data')
        
if(USE_NORM):
    scalers, norm_df = normalize(df)
    sequences = generate_sequences(norm_df, sequence_len, output_len, targets_list)
else:
    sequences = generate_sequences(df, sequence_len, output_len, targets_list)

#print(sequences[0]['target'])
#print(norm_df['2000-01-31':'2000-01-31'][targets_list].values)
dataset = SequenceDataset(sequences)

# Split the data according to our split ratio and load each subset into a
# separate DataLoader object
train_loader, val_loader = get_train_loader_splits(dataset, split, batch_size)

# Save state identifier
model_num = sys.argv[1]

# Initialize the model
model = LSTMForecaster(input_features, nhid, output_features, sequence_len, n_deep_layers=n_dnn_layers, use_cuda=USE_CUDA).to(device)
# Set learning rate and number of epochs to train over
lr = 4e-4
n_epochs = 30
# Initialize the loss function and optimizer
criterion = nn.MSELoss().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)



# Lists to store training and validation losses
t_losses, v_losses = [], []
# Loop over epochs
for epoch in range(n_epochs):
    train_loss, valid_loss = 0.0, 0.0

    # train step
    model.train()
    # Loop over train dataset
    for x, y, _ in train_loader:
        optimizer.zero_grad()
        # move inputs to device
        x = x.to(device)
        y  = y.squeeze().to(device)
        # Forward Pass
        
        preds = model(x).squeeze()
        loss = criterion(preds, y) # compute batch loss
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
    epoch_loss = train_loss / len(train_loader)
    t_losses.append(epoch_loss)
    
    # validation step
    model.eval()
    for x, y, _ in val_loader:
        with torch.no_grad():
            x, y = x.to(device), y.squeeze().to(device)
            #print(f"x shape = {x.shape}")
            #print(f"x = {x}")
            preds = model(x).squeeze()
            error = criterion(preds, y)
        valid_loss += error.item()
    valid_loss = valid_loss / len(val_loader)
    v_losses.append(valid_loss)
        
    print(f'{epoch} - train: {epoch_loss}, valid: {valid_loss}')
    path = f"../save_states/{model_num}_{epoch}.pth"
    torch.save(model.state_dict(), path)
best_epoch = np.argmin(v_losses)
print(f"Best Validation Loss: Epoch {best_epoch} with {np.min(v_losses)}")
for i in range(n_epochs):
    if(i != best_epoch):
        os.remove(f"../save_states/{model_num}_{i}.pth")

plot_losses(t_losses, v_losses)
