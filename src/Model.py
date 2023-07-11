from tensorflow import keras as ks
from tensorflow.keras.layers import GRU, Input, Dense, Flatten, Dropout, Reshape


class Generator:

    def __init__(self, input_shape, embedded_units, n_layers):

        self.input_shape = input_shape
        self.embedded_units = embedded_units
        self.n_layers = n_layers

# next we define our network parts
    def build_network_part(self):

        model = ks.models.Sequential(name="Generator")
        model.add(Input(shape=self.input_shape))
        for i in range(self.n_layers):
            model.add(GRU(units=self.embedded_units, return_sequences=True))
        model.add(Dense(units = self.embedded_units, activation = 'tanh'))

        return model

class Discriminator:

    def __init__(self, input_shape, hidden_units, n_layers, dropout_rate):

        self.input_shape = input_shape
        self.hidden_units = hidden_units
        self.n_layers = n_layers
        self.dropout_rate = dropout_rate

# next we define our network parts
    def build_network_part(self):

        model = ks.models.Sequential(name="Discriminator")
        model.add(Input(shape=self.input_shape))
        for i in range(self.n_layers):
            model.add(GRU(units=self.hidden_units, return_sequences=True))
            model.add(Dropout(self.dropout_rate))
        model.add(Flatten())
        model.add(Dense(units = 1, activation = 'sigmoid'))

        return model

class Embedder:

    def __init__(self, input_shape, embedded_units, n_layers):

        self.input_shape = input_shape
        self.embedded_units = embedded_units
        self.n_layers = n_layers

# next we define our network parts
    def build_network_part(self):

        model = ks.models.Sequential(name="Embedder")
        model.add(Input(shape=self.input_shape))
        for i in range(self.n_layers):
            model.add(GRU(units=self.embedded_units, return_sequences=True))
        model.add(Dense(units = self.embedded_units, activation = 'tanh'))

        return model

class Reconstructor:

    def __init__(self, input_shape, num_features, n_layers):

        self.input_shape = input_shape
        self.num_features = num_features
        self.n_layers = n_layers

# next we define our network parts
    def build_network_part(self):

        model = ks.models.Sequential(name="Reconstructor")
        model.add(Input(shape=self.input_shape))
        for i in range(self.n_layers):
            model.add(GRU(units=self.input_shape[1], return_sequences=True))
        model.add(Dense(units = self.input_shape[1], activation = 'tanh'))
        model.add(Dense(units = self.num_features))

        return model

class Supervisor:

    def __init__(self, input_shape, embedded_units, n_layers):

        self.input_shape = input_shape
        self.embedded_units = embedded_units
        self.n_layers = n_layers

# next we define our network parts
    def build_network_part(self):

        model = ks.models.Sequential(name="Supervisor")
        model.add(Input(shape=self.input_shape))
        for i in range(self.n_layers):
            model.add(GRU(units=self.embedded_units, return_sequences=True))
        model.add(Dense(units = self.embedded_units, activation = 'tanh'))

        return model


class TimeGAN:

    def __init__(self, seq_length, num_features, hidden_dims, n_layers):

        self.input_shape = input_shape
        self.hidden_units = hidden_units
        self.output_dims = output_dims
        self.n_layers = n_layers



