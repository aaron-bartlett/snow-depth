import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

dictionary = {
    "WT01" : "Fog",
    "WT02" : "Heavy Fog",
    "WT03" : "Thunder",
    "WT04" : "Small Hail",
    "WT05" : "Hail",
    "WT06" : "Glaze or Rime",
    "WT07" : "Blowing Dust",
    "WT08" : "Smoke or Haze",
    "WT09" : "Blowing Snow",
    "WT10" : "Tornado",
    "WT11" : "Damaging Winds",
    "WT13" : "Mist",
    "WT14" : "Drizzle",
    "WT15" : "Freezing Drizzle",
    "WT16" : "Rain",
    "WT17" : "Freezing Rain",
    "WT18" : "Snow",
    "WT19" : "Unknown Precipitation",
    "WT21" : "Ground Fog",
    "WT22" : "Freezing Fog",
    "AWND" : "Avg. Wind Speed (m/s)",
    "PRCP" : "Precipitation (mm)",
    "SNOW" : "Snow (mm)",
    "SNWD" : "Snow Depth (mm)",
    "TMAX" : "Max Temperature (deg C)",
    "TMIN" : "Min Temperature (deg C)", 
    "WDF2" : "Max Wind Direction (degrees)",
    "WSF2" : "Max Wind Speed (m/s)",
    "DOTY" : "Day of the Year",
    "Year" : "Year"
}


def initialize_db():
    data = pd.read_csv("dataset.csv", index_col="DATE")
    data.index = pd.to_datetime(data.index)
    data["Year"] = data.index.year
    data["DOTY"] = data.index.day_of_year
    return data[data["DOTY"] != 366]
    #return data

metrics_list = ["AWND","PRCP","SNOW","SNWD","TMAX","TMIN","WDF2","WSF2"]
features_list = ["Year","DOTY",
                 "AWND","PRCP","SNOW","SNWD","TMAX","TMIN","WDF2","WSF2",
                 "WT01","WT02","WT03","WT04","WT05"]

def generate_features():
    data = initialize_db()
    metrics = data[features_list].copy()
    return metrics['2000-01-01':'2018-12-30'].fillna(0), metrics['2000-01-02':'2018-12-31'].fillna(0), metrics['2019-01-01':'2023-11-17'].fillna(0), metrics['2019-01-02':'2023-11-18'].fillna(0)