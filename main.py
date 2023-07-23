import numpy as np

from src.timeGAN import TimeGAN
from tensorflow.keras.losses import BinaryCrossentropy, MeanSquaredError
from data_preprocessing import processing as p
from src import visualizations as v


if __name__ == '__main__':

    model_dimensions = {"seq_length": 5, "input_features": 10, "output_features": 10, "embedded_dims": 5}
    model_parameters = {"n_layers": 3, "mu": .5, "phi": .25, "lambda": 0.1, "alpha_1": 0.005, "alpha_2": 0.001,
                        "theta": 1}
    loss_functions = {"reconstruction_loss": MeanSquaredError(),
                      "supervised_loss": MeanSquaredError(),
                      "unsupervised_loss": BinaryCrossentropy()}
    batch_size = 3

    dates = p.import_data("SP500 raw.csv", True)[:16]
    data = p.import_data("SP500 raw.csv", False)[:16,:]
    data = p.transform_percent_change(data)
    sequences, scaler = p.generate_sequences(data, 3, True, 5)

    timeGAN = TimeGAN(model_dimensions, model_parameters, loss_functions, batch_size, scaler)
    print(timeGAN.get_summary())

    #feature_labels = ["Open","High","Low","Close","Volume"]

    #v.plot_generated_sequence(timeGAN, sequences, 0, feature_labels)

    #v.plot_autoencoded_sequence(timeGAN, sequences, 0, feature_labels, 100)




    """

    dates = p.import_data("SP500 raw.csv", True)[:16]
    data = p.import_data("SP500 raw.csv", False)[:16, :]
    data = p.transform_percent_change(data)

    print(data)
    print(np.shape(data))
    print(data[1:4,:])

    seq_length = 3
    feature_labels = ["Open","High","Low","Close","Volume"]

    v.plot_real_sequence(dates, data, feature_labels, seq_length)"""




