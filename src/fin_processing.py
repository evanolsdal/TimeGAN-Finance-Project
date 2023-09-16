from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd
import numpy as np

"""
new file for processing specifically financial data
"""

# This function loads the data placed in the data subfolder, where the path argument is the name of the data in that
# folder. The function loads in the csv into a pandas df, then converts this to np array and reverses the order so
# the data is chronologically ordered.
def import_fin_data(path, dates):

    if dates:
        return pd.read_csv(f"data/{path}").values[:,0][::-1]
    else:
        return pd.read_csv(f"data/{path}").values[:,1:][::-1]


# This function transforms the values into percent changes
def transform_percent_change(data):

    diff = np.diff(data, n=1, axis = 0)

    percentage = diff / data[:-1,:]

    return percentage

# This function creates sequenced data from the raw data and returns a numpy array
def generate_sequences(data, seq_len, scaler_type):

    scaler = None

    if scaler_type == 'minmax':
        scaler = MinMaxScaler(feature_range=(-1,1)).fit(data)
        data = scaler.transform(data)
    if scaler_type == 'normalize':
        scaler = StandardScaler().fit(data)
        data = scaler.transform(data)

    temp_data = []

    for i in range(len(data)-seq_len+1):

        x = data[i:i+seq_len,:]
        temp_data.append(x)

    return np.array(temp_data), scaler

# reverse the percentage differencing of the data, either pass a full sequence, or just
# a single instance
def reverse_sequences(raw_data, percentages, scaler, index=None):
    ind = None  # Initialize ind outside of the if statement

    if index is None:
        ind = np.random.randint(len(percentages))
    else:
        ind = index

    reverse_seq = percentages[ind]

    reverse_seq = scaler.inverse_transform(reverse_seq)

    reverse_seq[0] = raw_data[ind]

    for i in range(1, len(reverse_seq)):
        reverse_seq[i] = reverse_seq[i - 1] * (1 + reverse_seq[i])

    return reverse_seq


