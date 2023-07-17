import pandas as pd
import numpy as np
from sklearn.preprocessing import Normalizer

"""
This module creates the functions used for pre_processing the stock data into a format suitable for training
"""

# This function loads the data placed in the data subfolder, where the path argument is the name of the data in that
# folder. The function loads in the csv into a pandas df, then converts this to np array and reverses the order so
# the data is chronologically ordered.
def import_data(path, dates, volume_included):


    if dates:
        return pd.read_csv(f"data/{path}").values[:,0][::-1]
    elif volume_included:
        return pd.read_csv(f"data/{path}").values[:,1:][::-1]
    else:
        return pd.read_csv(f"data/{path}").values[:,1:-1][::-1]


# This function transforms the values into percent changes
def transform_percent_change(data):

    diff = np.diff(data, n=1, axis = 0)

    percentage = diff / data[:-1,:]

    return percentage


# This function creates sequenced data from the raw data and returns a numpy array
def generate_sequences(data, seq_len):


    scaler = Normalizer().fit(data)
    data = scaler.transform(data)

    temp_data = []

    for i in range(len(data)-seq_len+1):

        x = data[i:i+seq_len,:]
        temp_data.append(x)

    return np.array(temp_data)


