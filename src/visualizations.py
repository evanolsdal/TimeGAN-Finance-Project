import matplotlib.pyplot as plt

"""
This module deals with visualizing the training losses and the data sequences
"""

# Function for seeing the actual output

def plot_stock_trend(dates, data):

    # First turn dates into a list and then find the number of dimesnions in the data
    num_dimensions = data.shape[1]

    # Plot the trend for each dimension
    for dim in range(num_dimensions):
        plt.plot(dates[1:], data[:,dim].flatten(), label=f'Dimension {dim + 1}')

    plt.xlabel('Date')
    plt.ylabel('Percentage Change')
    plt.title('Trend in Percentage Change of Stock for Different Dimensions')
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.show()

