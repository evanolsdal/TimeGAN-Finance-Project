from src.timeGAN import TimeGAN
from tensorflow.keras.losses import BinaryCrossentropy, MeanSquaredError
from data_preprocessing import processing as p
from src.visualizations import plot_stock_trend


if __name__ == '__main__':

    model_dimensions = {"seq_length":3, "num_features":4, "embedded_dims":4}
    model_parameters = {"n_layers":2, "supervised_regularization":1}
    loss_functions = {"reconstruction_loss":MeanSquaredError(),
                      "supervised_loss":MeanSquaredError(),
                      "unsupervised_loss":BinaryCrossentropy()}
    batch_size = 5

    timeGAN = TimeGAN(model_dimensions, model_parameters, loss_functions, batch_size)

    dates = p.import_data("SP500 raw.csv", True, False)[:16]
    data = p.import_data("SP500 raw.csv", False, False)[:16,:]
    data = p.transform_percent_change(data)
    data = p.generate_sequences(data, 3)

    timeGAN.compile()

    #timeGAN.get_summary()

    #print(timeGAN.batch_data(data))

    reconstruction_loss = timeGAN.fit_autoencoder(data, 3)

    supervisor_loss = timeGAN.fit_autoencoder(data, 3)

    dynamic_losses = timeGAN.fit_dynamic_game(data, 3, 2)




