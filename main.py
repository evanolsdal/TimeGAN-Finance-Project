from src.timeGAN import TimeGAN
from tensorflow.keras.losses import BinaryCrossentropy, MeanSquaredError
from data_preprocessing import processing as p
from src.visualizations import plot_stock_trend


if __name__ == '__main__':

    model_dimensions = {"seq_length":3, "num_features":5, "embedded_dims":5}
    model_parameters = {"n_layers":2, "mu_g":.5, "mu_e":1, "lambda":.1, "alpha_1":0.001, "alpha_2":0.001}
    loss_functions = {"reconstruction_loss":MeanSquaredError(),
                      "supervised_loss":MeanSquaredError(),
                      "unsupervised_loss":BinaryCrossentropy()}
    batch_size = 5

    dates = p.import_data("SP500 raw.csv", True)[:16]
    data = p.import_data("SP500 raw.csv", False)[:16,:]
    data = p.transform_percent_change(data)
    data, normalizer = p.generate_sequences(data, 3)

    timeGAN = TimeGAN(model_dimensions, model_parameters, loss_functions, batch_size, normalizer)
    timeGAN.compile()

    #timeGAN.get_summary()

    #print(timeGAN.batch_data(data))

    autoencoder = timeGAN.fit_autoencoder(data, 3)

    dynamic_losses = timeGAN.fit_dynamic_game(data, 3, 2)

    print(dynamic_losses.get("Unsupervised Discriminator Loss"))




