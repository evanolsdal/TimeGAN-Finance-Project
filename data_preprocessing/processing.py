import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn import MinMaxScaler

"""
This module creates the functions used for pre_processing the stock data into a format suitable for training
"""

# This function loads the data placed in the data subfolder. The path argument is the name of the data in that folder
def import_data(path):

    return pd.read_csv(f"data/{path}").values

# This function creates sequenced data from the raw data and returns a numpy array
def generate_sequences(data, seq_len):

    data = data[::-1]

    scaler = MinMaxScaler().fit(data)
    data = scaler.transform(data)

    temp_data = []

    for i in range(len(data)-seq_len):

        x = data[i, i+seq_len]
        temp_data.append(x)

    return np.array(temp_data)
