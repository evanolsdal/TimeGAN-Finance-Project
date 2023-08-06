import numpy as np
from scipy.stats import pearsonr

"""
This module defines the functions to analyze the statistical properties of financial time series.
"""

# This function computes the three different correlations to be analyzed
def compute_autocorrelations(real_sequences, synthetic_sequences, dim, type):

    # find the sequence length
    seq_length = real_sequences.shape[1]

    # first select the dimension of interest
    real_sequences = np.squeeze(real_sequences[:,:,dim])
    synthetic_sequences = np.squeeze(synthetic_sequences[:,:,dim])

    # store the correlations
    real_correlations = []
    synthetics_correlations = []

    # get the first values in each sequence
    real_seq_first = np.squeeze(real_sequences[:, 0])
    synth_seq_first = np.squeeze(synthetic_sequences[:, 0])

    # then compute the correlation
    for i in range(seq_length-1):

        if type = "Linear Autocorrelation":

            # find the other sequences
            real_seq_other = np.squeeze(real_sequences[:, i+1])
            synth_seq_other = np.squeeze(synthetic_sequences[:, i+1])

            # compute the correlations
            real_corr, _ = pearsonr(real_seq_first, real_seq_other)
            synth_corr, _ = pearsonr(synth_seq_first, synth_seq_other)

            # add the correlations
            real_correlations.append(real_corr)
            synthetics_correlations.append(synth_corr)

        if type = "Volatility Clustering":

            # find the other sequences
            real_seq_other = np.squeeze(real_sequences[:, i+1])
            synth_seq_other = np.squeeze(synthetic_sequences[:, i+1])

            # compute the correlations
            real_corr, _ = pearsonr(np.abs(real_seq_first), np.abs(real_seq_other))
            synth_corr, _ = pearsonr(np.abs(synth_seq_first), np.abs(synth_seq_other))

            # add the correlations
            real_correlations.append(real_corr)
            synthetics_correlations.append(synth_corr)

        if type = "Leverage Effect":

            # find the other sequences
            real_seq_other = np.squeeze(real_sequences[:, i+1])
            synth_seq_other = np.squeeze(synthetic_sequences[:, i+1])

            # compute the correlations
            real_corr, _ = pearsonr(real_seq_first, real_seq_other*real_seq_other)
            synth_corr, _ = pearsonr(synth_seq_first, synth_seq_other*real_seq_other)

            # add the correlations
            real_correlations.append(real_corr)
            synthetics_correlations.append(synth_corr)

    return real_correlations, synthetics_correlations



