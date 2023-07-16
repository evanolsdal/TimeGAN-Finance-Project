# This is a sample Python script.
from src.timeGAN import TimeGAN
from tensorflow.keras.losses import BinaryCrossentropy, MeanSquaredError

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    model_dimensions = {"seq_length":25, "num_features":5, "embedded_dims":5}
    model_parameters = {"n_layers":2, "supervised_regularization":1}
    loss_functions = {"reconstruction_loss":MeanSquaredError,
                      "supervised_loss":MeanSquaredError,
                      "unsupervised_loss":BinaryCrossentropy}
    batch_size = 50

    timeGAN = TimeGAN(model_dimensions, model_parameters, loss_functions, batch_size)

    timeGAN.get_summary()





# See PyCharm help at https://www.jetbrains.com/help/pycharm/
