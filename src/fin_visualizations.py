import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fin_processing import reverse_sequences
import mplfinance as mpf

"""
A new set of graphing functions that more accurately display data relevant to stock sequences
"""

# plot sequences of output
def plot_seq(raw_data, sequences, scaler):
    ind = np.random.randint(len(sequences[0]))

    for i in range(len(sequences)):
        reversed_sequence = reverse_sequences(raw_data, sequences[i], scaler, index=ind)

        df = pd.DataFrame(reversed_sequence, columns=['Open', 'High', 'Low', 'Close'])

        # Plot a candlestick chart
        fig, ax = mpf.plot(df, type='candle', style='charles', title=f'Stock Price Progression - Sequence {i + 1}',
                           ylabel='Stock Price', volume=False, returnfig=True)

        plt.show()



