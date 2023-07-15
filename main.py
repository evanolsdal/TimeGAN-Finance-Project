# This is a sample Python script.
from src.networkparts import Generator, Discriminator, Embedder, Reconstructor, Supervisor

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    seq_length = 24
    num_features = 1
    hidden_dims = 3
    n_layers = 2

    generator = Generator((seq_length, num_features), hidden_dims, n_layers).build_network_part()
    generator.summary()

    supervisor = Supervisor((seq_length, num_features), hidden_dims, n_layers).build_network_part()
    supervisor.summary()

    embedder = Embedder((seq_length, num_features), hidden_dims, n_layers).build_network_part()
    embedder.summary()

    discriminator = Discriminator((seq_length, hidden_dims), hidden_dims, n_layers, .3).build_network_part()
    discriminator.summary()

    reconstructor = Reconstructor((seq_length, hidden_dims), num_features, n_layers).build_network_part()
    reconstructor.summary()



# See PyCharm help at https://www.jetbrains.com/help/pycharm/
