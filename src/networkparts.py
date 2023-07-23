from tensorflow import keras as ks
from tensorflow.keras.layers import GRU, Input, Dense, Flatten, Dropout, Reshape, Bidirectional, Activation, Rescaling

class Generator:

    def __init__(self, input_shape, embedded_units, n_layers, scale_tanh):

        self.input_shape = input_shape
        self.embedded_units = embedded_units
        self.n_layers = n_layers
        self.scale_tanh = scale_tanh


    # next we define our network parts
    def build_network_part(self):

        model = ks.models.Sequential(name="Generator")
        model.add(Input(shape=self.input_shape))
        for i in range(self.n_layers+1):
            model.add(GRU(units=self.embedded_units, return_sequences=True))
            model.add(Rescaling(scale=self.scale_tanh))
        model.add(Dense(units = self.embedded_units, activation='tanh'))
        model.add(Rescaling(scale=self.scale_tanh))

        return model

class Recovery:

    def __init__(self, input_shape, num_features, n_layers, scale_tanh):

        self.input_shape = input_shape
        self.n_layers = n_layers
        self.num_features = num_features
        self.scale_tanh = scale_tanh


    # next we define our network parts
    def build_network_part(self):

        model = ks.models.Sequential(name="Recovery")
        model.add(Input(shape=self.input_shape))
        for i in range(self.n_layers):
            model.add(GRU(units=self.num_features, return_sequences=True))
            model.add(Rescaling(scale = self.scale_tanh))
        model.add(Dense(units=self.num_features, activation = None))

        return model

class Embedder:

    def __init__(self, input_shape, embedded_units, n_layers, scale_tanh):

        self.input_shape = input_shape
        self.embedded_units = embedded_units
        self.n_layers = n_layers
        self.scale_tanh = scale_tanh


    # next we define our network parts
    def build_network_part(self):

        model = ks.models.Sequential(name="Embedder")
        model.add(Input(shape=self.input_shape))
        for i in range(self.n_layers):
            model.add(GRU(units=self.embedded_units, return_sequences=True))
            model.add(Rescaling(scale=self.scale_tanh))
        model.add(Dense(units = self.embedded_units, activation = 'tanh'))
        model.add(Rescaling(scale=self.scale_tanh))

        return model

class Discriminator:

    def __init__(self, input_shape, n_layers):

        self.input_shape = input_shape
        self.n_layers = n_layers

    # next we define our network parts
    def build_network_part(self):

        # maybe we will add less layers
        # new_n_layers = round(self.n_layers/2)

        new_n_layers = self.n_layers

        model = ks.models.Sequential(name="Discriminator")
        model.add(Input(shape=self.input_shape))
        for i in range(new_n_layers):
            model.add(Bidirectional(GRU(units=self.input_shape[1], return_sequences=True)))
        model.add(Flatten())
        model.add(Dense(units = 1, activation = 'sigmoid'))

        return model

class Supervisor:

    def __init__(self, input_shape, embedded_units, n_layers, scale_tanh):

        self.input_shape = input_shape
        self.embedded_units = embedded_units
        self.n_layers = n_layers
        self.scale_tanh = scale_tanh


    # next we define our network parts
    def build_network_part(self):

        model = ks.models.Sequential(name="Supervisor")
        model.add(Input(shape=self.input_shape))
        for i in range(self.n_layers):
            model.add(GRU(units=self.embedded_units, return_sequences=True))
            model.add(Rescaling(scale=self.scale_tanh))
        model.add(Dense(units = self.embedded_units, activation = 'tanh'))
        model.add(Rescaling(scale=self.scale_tanh))

        return model






