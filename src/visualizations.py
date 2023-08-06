import matplotlib.pyplot as plt
import numpy as np
from stat_properties import compute_autocorrelations

"""
This module deals with visualizing the training losses and the data sequences
"""

# Function for seeing the actual output

def plot_real_sequence(dates, data, labels, seq_length):

    # check if it's possible to slice
    max_length = len(data)
    if seq_length > max_length:
        raise ValueError("Desired sequence length exceeds the maximum length of the data.")

    # get a random slice to visualize
    start_index = np.random.randint(max_length - seq_length + 1)
    end_index = start_index + seq_length

    dates = dates[start_index:end_index]
    data = data[start_index:end_index, :]

    # get the number of dimensions
    num_dimensions = data.shape[-1]

    # Plotting the combined graph
    plt.figure(figsize=(10, 6))
    for dim in range(num_dimensions):
        plt.plot(dates, data[:, dim], label=labels[dim])
        plt.axhline(y=0, color='red', linestyle='--')
    plt.title('Data with Multiple Dimensions')
    plt.xlabel('Dates')
    plt.ylabel('Value')
    plt.xticks(rotation=45)
    plt.legend()
    plt.show()

    # Plotting individual subgraphs
    fig, axs = plt.subplots(num_dimensions, 1, figsize=(10, 8))
    for dim in range(num_dimensions):
        axs[dim].plot(dates, data[:, dim])
        axs[dim].set_title(labels[dim])
        axs[dim].axhline(y=0, color='red', linestyle='--')
        axs[dim].set_xlabel('Dates')
        axs[dim].set_ylabel('Value')
        axs[dim].tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.show()

# function that plots a generated and autoencoded sequence on the same graph
def plot_generated_sequence(model, sequences, feature, feature_labels):

    # get the generated and autoencoded sequence
    generated_seq = model.generate_seq(1)
    seq = sequences[np.random.randint(len(sequences)),:,:]
    autoencoded_seq = model.autoencode_seq(seq)

    # select the dimension feature from the two sequences
    generated_seq = generated_seq[:, feature]
    autoencoded_seq = autoencoded_seq[:,feature]

    # Plotting the combined graph
    plt.figure(figsize=(8, 6))
    plt.plot(generated_seq, label='Generated')
    plt.plot(autoencoded_seq, label='Autoencoded')
    plt.axhline(y=0, color='red', linestyle='--')
    plt.title(f"Generated and Autoencoded Sequences for {feature_labels[feature]}")
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.show()

    # Plotting individual subgraphs
    fig, axs = plt.subplots(2, 1, figsize=(8, 8))
    axs[0].plot(autoencoded_seq)
    axs[0].axhline(y=0, color='red', linestyle='--')
    axs[0].set_title(f"Autoencoded Sequences for {feature_labels[feature]}")
    axs[0].set_xlabel('Time')
    axs[0].set_ylabel('Value')

    axs[1].plot(generated_seq)
    axs[1].axhline(y=0, color='red', linestyle='--')
    axs[1].set_title(f"Generated Sequences for {feature_labels[feature]}")
    axs[1].set_xlabel('Time')
    axs[1].set_ylabel('Value')

    plt.tight_layout()
    plt.show()

# function that plots the real and corresponding autoencoded sequence on the same graph
def plot_autoencoded_sequence(model, sequences, feature, feature_labels):

    # draw the real and encoded sequence
    real_seq = sequences[np.random.randint(len(sequences)), :, :]
    autoencoded_seq = model.autoencode_seq(real_seq)

    # select the dimension feature from the two sequences and scale up autoencoded sequence
    real_seq = real_seq[:, feature]
    autoencoded_seq = autoencoded_seq[:, feature]

    # Plotting the combined graph
    plt.figure(figsize=(8, 6))
    plt.plot(real_seq, label='Real Sequence')
    plt.plot(autoencoded_seq, label='Autoencoded Sequence')
    plt.axhline(y=0, color='red', linestyle='--')
    plt.title(f"Real and Autoencoded Sequences for {feature_labels[feature]}")
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.show()

    # Plotting individual subgraphs
    fig, axs = plt.subplots(2, 1, figsize=(8, 8))
    axs[0].plot(real_seq)
    axs[0].axhline(y=0, color='red', linestyle='--')
    axs[0].set_title(f"Real Sequences for {feature_labels[feature]}")
    axs[0].set_xlabel('Time')
    axs[0].set_ylabel('Value')

    axs[1].plot(autoencoded_seq)
    axs[1].axhline(y=0, color='red', linestyle='--')
    axs[1].set_title(f"Autoencoded Sequences for {feature_labels[feature]}")
    axs[1].set_xlabel('Time')
    axs[1].set_ylabel('Value')

    plt.tight_layout()
    plt.show()

# function that plots the losses on the same graph
def plot_losses(losses):

    # Plotting the combined graph
    plt.figure(figsize=(10, 6))
    for key, values in losses.items():
        plt.plot(values, label=key)
    plt.title('All Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Plotting individual subgraphs
    fig, axs = plt.subplots(len(losses), 1, figsize=(10, 6))
    for i, (key, values) in enumerate(losses.items()):
        axs[i].plot(values)
        axs[i].set_title(key)
        axs[i].set_xlabel('Epoch')
        axs[i].set_ylabel('Loss')
    plt.tight_layout()
    plt.show()

# function that plots the autocorrelation statistics
def plot_autocorrelations(real_sequences, synthetic_sequences, dim_labels, dim, type):

    # compute the correlations
    real_correlations, synthetic_correlations = compute_autocorrelations(real_sequences, synthetic_sequences, dim, type)
    lags = np.arange(len(real_correlations))

    plt.figure(figsize=(10, 6))

    # Plot real autocorrelations
    plt.subplot(2, 1, 1)
    plt.plot(lags, real_correlations)
    plt.title("Real " + type + " for " + dim_labels[dim])
    plt.ylabel("Correlation")
    plt.xlabel("Lag")

    # Plot synthetic autocorrelations
    plt.subplot(2, 1, 2)
    plt.plot(lags, synthetic_correlations)
    plt.title("Synthetic " + type + " for " + dim_labels[dim])
    plt.ylabel("Correlation")
    plt.xlabel("Lag")

    plt.tight_layout()
    plt.show()



