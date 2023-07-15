from tensorflow.keras.models import Model
import tensorflow as tf
from src import networkparts
import numpy as np

# This uses the model

class TimeGAN(Model):

    def __init__(self, dimensions, parameters, reconstruction_loss, supervised_loss, unsupervised_loss, batch_size, *args, **kwargs):

        #superimpose the other init args
        super().__init__(*args, **kwargs)

        # add the dimension arguments
        self.dimensions = dimensions
        self.parameters = parameters
        self.batch_size = batch_size
        self.reconstruction_loss = reconstruction_loss
        self.supervised_loss = supervised_loss
        self.unsupervised_loss = unsupervised_loss

        # extract important parameters for network part construction
        seq_len = dimensions.get(seq_length)
        features = dimensions.get(num_features)
        embed = dimensions.get(embedded_dims)
        n = dimensions.get(n_layers)
        droput = parameters.get(dropout_rate)

        # then add the network parts
        self.generator = networkparts.Generator((seq_len, features), embed, n).build_network_part()
        self.discriminator = networkparts.Discriminator((seq_len, embed), droput, n).build_network_part()
        self.embedder = networkparts.Embedder((seq_len, features), embed, n).build_network_part()
        self.reconstructor = networkparts.Recovery((seq_len, embed), features, n).build_network_part()

    # Defines the loss functions used for the training of the model
    def compile(self, *args, **kwargs):

        super().compile(*args, **kwargs)

    # generates input noise for the generator
    def get_noise(self):

        return tf.random.normal((batch, self.dimensions.get(seq_length), self.dimensions.get(num_features)))

    def train_step(self, batch):

        """
        This is defines the whole outer loop used for training. The step trains the networks parts in the
        following order:
        1) Embedder - reconstruciton loss
        2) Embedder + Generator - supervised loss
        3) Generator - unsupervised loss
        4) Discriminator - unsupervised loss
        5) Recovery - reconstruction loss
        This ensures that the corresponding network parts are being updated in the correct order (ex the
        Recovery function is optimized for the newly trained Embedder).
        """

        # first we generate the input noise for the generator and the real data batch
        X = batch
        Z = self.get_noise()

        # train the embedder based on reconstructor loss
        with tf.GradientTape() as e_tape:

            # compute the recovered data
            E = self.embedder(X, traning = True)
            X_hat = self.reconstructor(E, training = False)

            # compute the loss
            embedded_loss = self.reconstruction_loss(X, X_hat)

        # compute and apply the gradient
        e_grad = e_tape.gradient(embedded_loss, self.embedder.trainable_variables)
        tf.keras.optimizers.Adam().apply_gradients(zip(e_grad, self.embedder.trainable_variables))

        # train the embedder and generator to minimize the supervised loss
        with tf.GradientTape() as supervised_tape:

            # compute the recovered data
            E = self.embedder(X, traning=True)
            E_hat = self.generator(Z, training=True)

            # compute the loss
            supervised_loss = self.supervised_loss(E, E_hat)

        # compute and apply the gradient
        supervised_variables = self.embedder.trainable_variables + self.generator.trainable_variables
        supervised_grad = supervised_tape.gradient(supervised_loss, supervised_variables)
        tf.keras.optimizers.Adam().apply_gradients(zip(supervised_grad, supervised_variables))

        with tf.GradientTape() as g_tape:

            # define the
            E_hat = self.generator(Z, training=True)


            


