import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
torch.manual_seed(0)

from lstm import LSTMForecaster
from utils import *

if('-h' in sys.argv):
    print("Usage: python test_lstm.py <model_num> <epoch_num> <YYYY-MM-DD> [-n, -h]")
    sys.exit()

# Device selection (CPU | GPU)
USE_CUDA = torch.cuda.is_available()
device = 'cuda' if USE_CUDA else 'cpu'
print(f"Using '{device}' device")



df = pd.read_csv("../data/large_dataset.csv", index_col="DATE")
df.index = pd.to_datetime(df.index)
features_list = ["AWND","PRCP","SNOW","SNWD","TMAX","TMIN","WDF2","WSF2"]
targets_list = ["SNWD"]
#df["WSFX"] = df[['WSF1', 'WSF2']].max(axis=1)
#df["WDFX"] = df[['WDF1', 'WDF2']].max(axis=1)
df = df[features_list].copy()
df = df.fillna(0)

PATH = "../save_states/001_0.pth"
USE_NORM = False
if(len(sys.argv) > 3):
    PATH = f"../save_states/{sys.argv[1]}_{sys.argv[2]}.pth"
    if('-n' in sys.argv):
        USE_NORM = True
        print("Using Normalized Data")
else:
    print("Usage: python test_lstm.py <model_num> <epoch_num> <YYYY-MM-DD> [-n, -h]")

if(USE_NORM):
    scalers, norm_df = normalize(df)
    sequences = generate_sequences(norm_df, sequence_len, output_len, targets_list)
else:
    sequences = generate_sequences(df, sequence_len, output_len, targets_list)


dataset = SequenceDataset(sequences)

date = pd.to_datetime(sys.argv[3])
date_i = df.index.get_loc(date)


# Split the data according to our split ratio and load each subset into a
# separate DataLoader object
val_loader, test_loader = get_test_loader(dataset, date_i, split, batch_size)
full_loader = get_full_loader(dataset)

model = LSTMForecaster(input_features, nhid, output_features, sequence_len, n_deep_layers=n_dnn_layers, use_cuda=USE_CUDA).to(device)

model.eval()

model.load_state_dict(torch.load(PATH))

y_pred, y_true, y_dates = make_predictions_from_dataloader(model, test_loader)

print(f"PRED: {y_pred[0:10]}")
print(f"TRUE: {y_true[0:10]}")
print(f"DATES: {y_dates[0:10]}")

plot_preds(y_pred,y_true,y_dates)


# PLOT FULL DATASET

#y_pred, y_true, y_dates = make_predictions_from_dataloader(model, full_loader)
#plot_preds(y_pred,y_true,y_dates)



y_pred, y_true, y_dates = make_predictions_from_dataloader(model, val_loader)

print(f"Mean Error: {np.mean(y_true-y_pred)}")

print(f"MAE: {np.round(mean_absolute_error(y_true, y_pred),decimals=4)}")
print(f"MSE: {np.round(mean_squared_error(y_true, y_pred),decimals=4)}")

nzi = np.where(y_true != 0)[0] #Non-Zero Indices
print(f"Non-Zero MAE: {mean_absolute_error(y_true[nzi], y_pred[nzi])}")
print(f"Non-Zero MSE: {mean_squared_error(y_true[nzi], y_pred[nzi])}")

plot_matrix(y_pred, y_true)

num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total Parameters: {num_params}")



if(USE_NORM):
    date_features, date_true = get_date_sequence(norm_df, features_list, date, sequence_len)
else:
    date_features, date_true = get_date_sequence(df, features_list, date, sequence_len)

date_features = torch.tensor(date_features, dtype=torch.float32, requires_grad=True).unsqueeze(0)
#print(date_features.shape)
#print(date_features)

date_pred = model(date_features).squeeze()
print(f"DATE: {date.strftime('%Y-%m-%d')}, TRUE SNOW DEPTH: {date_true}")
print(f"PREDICTED SNOW DEPTH: {date_pred}")
graph_gradients(model)


