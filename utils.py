import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler, StandardScaler
from datetime import datetime, timedelta

import torch
import torch.nn as nn
torch.manual_seed(44)
from torch.utils.data import Dataset, DataLoader, random_split, Subset

nhid = 20 # Number of nodes in the hidden layer
n_dnn_layers = 1 # Number of hidden fully connected layers
input_features = 8
output_features = 1
batch_size = 16 # Training batch size
split = 0.8 # Train/Test Split ratio
sequence_len = 30
output_len = 1

class SequenceDataset(Dataset):

    def __init__(self, df):
        self.data = df

    def __getitem__(self, idx):
        sample = self.data[idx]
        index = sample['index'].to_numpy().astype('datetime64[D]')
        index = index.astype(datetime)
        index = int(index[0].strftime('%Y%m%d'))
        #print(type(index))
        #print(f"O: {index}")
        #index = index // 100000000000
        #print(f"N: {index}")
        return torch.Tensor(sample['sequence']), torch.Tensor(sample['target']), index
    
    def __len__(self):
        return len(self.data)
    
def normalize(df):
    # Fit scalers
    scalers = {}
    for x in df.columns:
        scalers[x] = RobustScaler().fit(df[x].values.reshape(-1, 1))


    # Transform data via scalers
    norm_df = df.copy()
    for i, key in enumerate(scalers.keys()):
        norm = scalers[key].transform(norm_df.iloc[:, i].values.reshape(-1, 1))
        norm_df.iloc[:, i] = norm
    return scalers, norm_df

# Defining a function that creates sequences and targets as shown above
def generate_sequences(df: pd.DataFrame, tw: int, pw: int, target_columns, drop_targets=False):
    '''
    df: Pandas DataFrame of the univariate time-series
    tw: Training Window - Integer defining how many steps to look back
    pw: Prediction Window - Integer defining how many steps forward to predict

    returns: dictionary of sequences and targets for all sequences
    '''
    data = dict() # Store results into a dictionary
    L = len(df)
    for i in range(L-tw):
        # Option to drop target from dataframe
        if drop_targets:
            df.drop(target_columns, axis=1, inplace=True)

        # Get current sequence  
        sequence = df[i:i+tw].values
        # Get values right after the current sequence
        target = df[i+tw:i+tw+pw][target_columns].values
        # Get datetime64 values for target sequence
        index = df[i+tw:i+tw+pw].index
        #print(f"TARGET: {target}")
        #print(f"INDEX: {index}")
        data[i] = {'sequence': sequence, 'target': target, 'index': index}

    return data

def get_train_loader_splits(dataset, split, batch_size):
    train_len = int(len(dataset)*split)
    lens = [train_len, len(dataset)-train_len]
    train_ds, val_ds = random_split(dataset, lens)
    trainloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    valloader = DataLoader(val_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    return trainloader, valloader

def get_test_loader(dataset, date, split, batch_size):
    test_indices = range(date-30, date + 70)
    test_ds = Subset(dataset, test_indices)
    testloader = DataLoader(test_ds)

    train_len = int(len(dataset)*split)
    lens = [train_len, len(dataset)-train_len]
    _, val_ds = random_split(dataset, lens)
    valloader = DataLoader(val_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    #testloader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=True)
    return valloader, testloader

def get_full_loader(dataset):
    return DataLoader(dataset)

def make_predictions_from_dataloader(model, unshuffled_dataloader):
    model.eval()
    predictions, actuals, dates = [], [], []
    for x, y, day in unshuffled_dataloader:
        with torch.no_grad():
            p = model(x)
            predictions.append(p)
            actuals.append(y)
            dates.append(day)
    predictions = torch.cat(predictions).numpy()
    actuals = torch.cat(actuals).numpy()
    dates = torch.cat(dates).numpy()
    return predictions.squeeze(), actuals.squeeze(), dates.squeeze()


def plot_losses(train_losses, val_losses):

    plt.figure(figsize=(10, 6))
    epochs = range(len(train_losses))
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')

    # Adding labels and title
    plt.xlabel('Epochs')
    plt.ylabel('Losses')
    plt.title('Training Loss Data')
    plt.legend()
    plt.savefig("figures/training_loss.png")


def plot_preds(y_pred, y_true, y_dates):

    sorted_i = np.argsort(y_dates)
    y_days = []
    for i in range(len(y_dates)):
        #print(datetime.strptime(str(y_dates[i]), '%Y%m%d'))
        y_days.append(datetime.strptime(str(y_dates[i]), '%Y%m%d'))
    y_pred = y_pred[sorted_i]
    y_true = y_true[sorted_i]
    y_days = np.array(y_days)
    y_days = y_days[sorted_i]

    plt.figure(figsize=(10,6))
    #time = np.arange(len(y_pred))
    #print(type(time[0]))
    #print(type(y_pred[0]))
    plt.plot(y_days, y_pred, 'r-', label='Predicted Values')
    plt.plot(y_days, y_true, 'g-', label='True Values')
    plt.xlabel('Day')
    plt.ylabel('Snow Depth (inches)')
    plt.title('True vs Predicted Snow Depth over given dates')
    plt.legend()
    plt.savefig("figures/prediction_gap.png")

def plot_matrix(y_pred, y_true):

    plt.figure(figsize=(10,6))
    plt.scatter(y_true, y_pred, s=5)
    limit = np.max(y_true)
    plt.plot([0, limit], [0, limit], 'r-')
    plt.xlabel('True Snow Depth')
    plt.ylabel('Predicted Snow Depth')
    plt.title('Predictions vs True Values of Snow Depth')

    plt.savefig("figures/prediction_grid.png")

def plot_error_areas(y_pred, y_true):
    y_err = y_true - y_pred
    plt.figure(10,6)
    plt.plot(y_true,y_err)
    plt.xlabel('True Snow Depth')
    plt.ylabel('Error')
    plt.title('Error by True Depth')

def get_date_sequence(df, features_list, date, sequence_len):
    s_date = date - timedelta(days=sequence_len)
    e_date = date - timedelta(days=1)
    sequence = df[s_date:e_date][features_list].values
    target = df[date:date]["SNWD"].values
    return sequence, target[0]

def graph_gradients(model):

    lstm_weights = []
    for param in model.parameters():
        lstm_weights.append(param.data.numpy())
    input_to_hidden_weights = lstm_weights[0]
    weight_magnitudes = np.abs(input_to_hidden_weights)
    average_magnitudes = np.mean(weight_magnitudes, axis=0)


    plt.figure(figsize=(10,6))
    plt.bar(features_list, average_magnitudes)
    plt.xlabel('Input Features')
    plt.ylabel('Average Magnitude of Weights')
    plt.title("Principal Component Analysis of Input Features")
    plt.savefig("figures/feature_magnitudes.png")

features_list = ["AWND","PRCP","SNOW","SNWD","TMAX","TMIN","WDFX","WSFX"]

features_dict = {
    "AWND": "Average Wind (mph)",
    "PRCP": "Precipitation (in)",
    "SNOW": "Snowfall (in)",
    "SNWD": "Snow Depth (in)",
    "TMAX": "Max Temperature (F)",
    "TMIN": "Min Temperature (F)",
    "WDF2": "Direction of Max Wind (degrees)",
    "WSF2": "Speed of Max Wind (mph)"
}