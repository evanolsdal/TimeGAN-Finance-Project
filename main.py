import numpy as np

from src.timeGAN import TimeGAN
from tensorflow.keras.losses import BinaryCrossentropy, MeanSquaredError
from data_preprocessing import processing as p
from src import visualizations as v


if __name__ == '__main__':

    model_dimensions = {"seq_length":3, "num_features":5, "embedded_dims":5}
    model_parameters = {"n_layers":2, "mu":.5, "phi":1, "lambda":.1, "alpha_1":0.001, "alpha_2":0.001, "pi":1}
    loss_functions = {"reconstruction_loss":MeanSquaredError(),
                      "supervised_loss":MeanSquaredError(),
                      "unsupervised_loss":BinaryCrossentropy()}
    batch_size = 5

    dates = p.import_data("SP500 raw.csv", True)[:16]
    data = p.import_data("SP500 raw.csv", False)[:16,:]
    data = p.transform_percent_change(data)
    sequences, scaler = p.generate_sequences(data, 3)

    timeGAN = TimeGAN(model_dimensions, model_parameters, loss_functions, batch_size, scaler)
    timeGAN.compile()

    feature_labels = ["Open","High","Low","Close","Volume"]

    #v.plot_generated_sequence(timeGAN, sequences, 0, feature_labels)

    #v.plot_autoencoded_sequence(timeGAN, sequences, 0, feature_labels, 100)

    print(sequences)
    print(timeGAN.get_noise(5))



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




