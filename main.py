from src.timeGAN import TimeGAN
from tensorflow.keras.losses import BinaryCrossentropy, MeanSquaredError
from data_preprocessing import processing as p
from src.visualizations import plot_stock_trend


if __name__ == '__main__':

    """model_dimensions = {"seq_length":25, "num_features":5, "embedded_dims":5}
    model_parameters = {"n_layers":2, "supervised_regularization":1}
    loss_functions = {"reconstruction_loss":MeanSquaredError,
                      "supervised_loss":MeanSquaredError,
                      "unsupervised_loss":BinaryCrossentropy}
    batch_size = 50

    timeGAN = TimeGAN(model_dimensions, model_parameters, loss_functions, batch_size)

    timeGAN.get_summary()"""

    dates = p.import_data("SP500 raw.csv", True, False)[:15]
    data = p.import_data("SP500 raw.csv", False, False)[:15,:]
    data = p.transform_percent_change(data)

    print(data)

    plot_stock_trend(dates, data)


