from tensorflow.keras.models import Model
import tensorflow as tf
from src import networkparts

"""
This module defines the TimeGAN model. 

Inputs:
    - model_dimensions: a dict containing all of the dimension needed for the model, namely
        - seq_length: length of time steps
        - num_features: number of features associated with each time step
        - embedded_dims: number of dimensions desired for hidden layers
    - model_parameters: a dict containing all of the parameters used to set up the model
        - n_layers: number of layers desired for each time step
        - supervised_regularization: regularizes the supervised loss in the generator loss function
    - loss_funcitons: a dict containing all of the loss functions used for training
        - reconsruction_loss: loss for the autoencoder 
        - supervised_loss: supervised loss function
        - unsupervised_loss: unsupervised loss function
    - batch_size: batch size for training
"""

class TimeGAN(Model):

    def __init__(self, model_dimensions, model_parameters, loss_functions, batch_size, *args, **kwargs):

        #superimpose the other init args
        super().__init__(*args, **kwargs)

        # add the dimension arguments
        self.model_dimensions = model_dimensions
        self.model_parameters = model_parameters
        self.batch_size = batch_size
        self.reconstruction_loss = loss_functions.get("reconstruction_loss")
        self.supervised_loss = loss_functions.get("supervised_loss")
        self.unsupervised_loss = loss_functions.get("unsupervised_loss")

        # extract important parameters for network part construction
        seq_len = model_dimensions.get("seq_length")
        features = model_dimensions.get("num_features")
        embed = model_dimensions.get("embedded_dims")
        n = model_parameters.get("n_layers")

        # then add the network parts
        self.generator = networkparts.Generator((seq_len, features), embed, n).build_network_part()
        self.discriminator = networkparts.Discriminator((seq_len, embed), n).build_network_part()
        self.supervisor = networkparts.Supervisor((seq_len, embed), embed, n).build_network_part()
        self.embedder = networkparts.Embedder((seq_len, features), embed, n).build_network_part()
        self.recovery = networkparts.Recovery((seq_len, embed), features, n).build_network_part()

    # Defines the loss functions used for the training of the model
    def compile(self, *args, **kwargs):

        super().compile(*args, **kwargs)

    # Summarizes the model of all the network parts
    def get_summary(self):

        print(self.generator.summary())
        print("###################################################################")
        print(self.discriminator.summary())
        print("###################################################################")
        print(self.supervisor.summary())
        print("###################################################################")
        print(self.embedder.summary())
        print("###################################################################")
        print(self.recovery.summary())


    ####################################################################################################################
    # This part defines the gradient steps for each network part to be used later in full training
    ####################################################################################################################

    """
    This trains the embedder and recovery function simultaneously for the encoded representation of the real data. The
    step aims to minimize the euclidian distance between the real data and the data recovered from the embedder. This
    ensures both that an effective embedded representation has been found, as well as an effective way of recovering the 
    data from this embedded represent has been found (so samples from the generator can be recovered after training).
    """
    def train_autoencoder_step(self, batch):

        X = batch

        with tf.GradientTape() as tape:
            # compute the recovered data
            E = self.embedder(X, traning=True)
            X_hat = self.recovery(E, training=True)

            # compute the loss
            reconstruction_loss = self.reconstruction_loss(X, X_hat)


        # compute and apply the gradient
        trainable_variables = self.embedder.trainable_variables + self.recovery.trainable_variables
        grad = tape.gradient(reconstruction_loss, trainable_variables)
        tf.keras.optimizers.Adam().apply_gradients(zip(grad, trainable_variables))

        return reconstruction_loss

    """
    This trains the supervisor to learn the stepwise dynamics of the real data in the embedded space. The stepwise
    dynamics are learned by minimizing the distance between it's output and the output of the next timestep in the
    real data. This way the supervisor learns how steps within the real data are taken. 
    """
    def train_supervisor_step(self, batch):

        X = batch

        with tf.GradientTape() as tape:
            # compute the recovered data
            E = self.embedder(X, traning=False)
            E_supervised = self.supervisor(E, training = False)

            # compute the loss
            supervised_loss = self.supervised_loss(E[:,1:,:], E_supervised[:,:-1,:])


        # compute and apply the gradient
        trainable_variables = self.supervisor.trainable_variables
        grad = tape.gradient(supervised_loss, trainable_variables)
        tf.keras.optimizers.Adam().apply_gradients(zip(grad, trainable_variables))

        return supervised_loss

    """
    This trains the generator in the dynamic game with the discriminator. It updates the generator in two stages, first
    minimizing the supervised loss, then minimizing the unsupervised loss. It breaks down as follows:
    
    1) First, the generator output is fed through the supervisor, and the new supervised loss is 
    computed with respect to the generator output. Since the supervisor is trained to mimic the stepwise dynamics of 
    the real data, this new loss forces the generators representation to adhere to those same stepwise dynamics by
    minimizing the distance between what the supervisor thinks the next timestep would be if the data was real,
    and the generators actual next timestep.
    2) Second, the generator output is fed to the discriminator, generating a label for whether or not the discriminator
    thinks this output is real. This output is then given the target label corresponding to real data (zero in this case),
    and the binary cross entropy is computed as the loss. By minimizing this loss, the generator is rewarded for getting
    the discriminator to label it's output as real.
    """
    def train_generator_step(self):

        Z = self.get_noise(self.batch_size)

        with tf.GradientTape() as tape:

            # generate the fake data and the supervised sequence of that data
            E_hat = self.generator(Z, traning = True)
            E_hat_supervised = self.supervisor(E_hat, training = False)

            # compute the supervised loss
            supervised_loss = self.supervised_loss(E_hat[:,1:,:], E_hat_supervised[:,:-1,:])

            # get the discriminator labels for the generators data and make corresponding target same as real data
            Y_hat = self.discriminator(E_hat, training = False)
            Y = tf.zeros_like(Y_hat)

            # compute the unsupervised loss
            unsupervised_loss = self.unsupervised_loss(Y, Y_hat)

            # add the supervised and unsupervised loss adding a dampening parameter to the supervised loss
            generator_loss = unsupervised_loss + self.model_parameters.get("supervised_regularization")* supervised_loss


        # compute and apply the gradient
        trainable_variables = self.generator.trainable_variables
        grad = tape.gradient(generator_loss, trainable_variables)
        tf.keras.optimizers.Adam().apply_gradients(zip(grad, trainable_variables))

        return supervised_loss, unsupervised_loss

    """
    This trains the discriminator in the dynamic game with the generator. The discriminator gets fed real embedded
    data and generated data, labels the two, and minimizes the binary cross entropy between the discriminator labels
    and the real labels.
    """
    def train_discriminator_step(self, batch):

        X = batch
        Z = self.get_noise(self.batch_size)

        with tf.GradientTape() as tape:

            # Generate the discriminator outputs
            E = self.embedder(X, training = False)
            E_hat = self.generator(Z, training = False)
            Y_hat_real = self.discriminator(E, training = True)
            Y_hat_fake = self.discriminator(E_hat, training = True)
            Y_hat = tf.concat(Y_hat_real, Y_hat_fake)

            # Create the real labels for the discriminator outputs
            Y = tf.concat(tf.zeros_like(Y_hat_real), tf.ones_like(Y_hat_fake))

            # compute the unsupervised discriminator loss
            discriminator_loss = self.unsupervised_loss(Y, Y_hat)


        # compute and apply the gradient
        trainable_variables = self.discriminator.trainable_variables
        grad = tape.gradient(discriminator_loss, trainable_variables)
        tf.keras.optimizers.Adam().apply_gradients(zip(grad, trainable_variables))


    ####################################################################################################################
    # Below are the training functions for each individual part of the network and the full fit functions.
    # Running the fit for individual parts of the network allows the user to optimize the training for each part
    # of the network separately, without having to train the full network.
    ####################################################################################################################

    """
    This runs the full training for the autoencoder
    """
    def fit_autoencoder(self, x_train, epochs):

        print(f"Starting Autoencoder Training")

        # generate the initial batched data set
        batched_data = self.batch_data(x_train)

        # create empty array to store the loss at each step of each epoch
        losses = []

        # iterate over the data epochs number of times
        for epoch in range(epochs):

            # reshuffle the data in each epoch
            batched_data = batched_data.shuffle(buffer_size=len(x_train))

            # create an empty array for the loss at each step in epoch
            epoch_losses = []

            # iterate over all batches in the batched data set
            for step, batch in enumerate(batched_data):

                # train the model and return the reconstruction loss
                reconstruction_loss = self.train_autoencoder_step(batch)

                # append the loss
                epoch_losses.append(reconstruction_loss)

                print(f"Epoch {epoch}, step {step}: Reconstruction loss = {reconstruction_loss}")

            losses.append(epoch_losses)

        print(f"Finished Autoencoder Training")

        return losses


    """
    This runs the full training for the supervisor
    """
    def fit_supervisor(self, x_train, epochs):

        print(f"Starting Supervisor Training")

        # generate the initial batched data set
        batched_data = self.batch_data(x_train)

        # create empty array to store the loss at each step of each epoch
        losses = []

        # iterate over the data epochs number of times
        for epoch in range(epochs):

            # reshuffle the data in each epoch
            batched_data = batched_data.shuffle(buffer_size=len(x_train))

            # create an empty array for the loss at each step in epoch
            epoch_losses = []

            # iterate over all batches in the batched data set
            for step, batch in enumerate(batched_data):
                # train the model and return the reconstruction loss
                supervised_loss = self.train_supervisor_step(batch)

                # append the loss
                epoch_losses.append(supervised_loss)

                print(f"Epoch {epoch}, step {step}: Supervised loss = {supervised_loss}")

            losses.append(epoch_losses)

        print(f"Finished Supervisor Training")

        return losses

    """
    This runs the full training for the dynamic game. This is a little more involved. First the generator is trained 
    for k steps, to allow for stronger data generation, then the discriminator is trained for 1 step. And this is done 
    for each epoch.
    """
    def fit_dynamic_game(self, x_train, epochs, k):

        print(f"Starting Dynamic Training")

        # generate the initial batched data set
        batched_data = self.batch_data(x_train)

        # create empty array to store the loss at each step of each epoch
        losses = []

        # iterate over the data autoencoder_epochs number of times
        for epoch in range(epochs):

            # reshuffle the data in each epoch
            batched_data = batched_data.shuffle(buffer_size=len(x_train))

            # create an empty array for the loss at each step in epoch
            epoch_losses = []

            # iterate over all batches in the batched data set
            for step, batch in enumerate(batched_data):

                # we then train the generator k times at each step

                # first create a temporary holder for the generator losses
                gen_s_loss, gen_u_loss = 0

                for i in range(k):

                    # train the generator and return the supervised and unsupervised losses
                    gen_s_loss, gen_u_loss = self.train_generator_step()

                # then train the discriminator
                disc_u_loss = self.train_discriminator_step(batch)

                # append the three losses to the epoch losses
                epoch_losses.append([gen_s_loss, gen_u_loss, disc_u_loss])

                print(f"Epoch {epoch}, step {step}: Generator S loss = {gen_s_loss}, Generator U loss = {gen_u_loss}, Discriminator U loss = {disc_u_loss}")

            losses.append(epoch_losses)

        print(f"Finished Dynamic Training")

        return losses


    """
    Finally define a fit function for training the whole TimeGAN
    """
    def fit(self, x_train, autoencoder_epochs, supervisor_epochs, dynamic_epochs, k):

        autoencoder_losses = self.fit_autoencoder(x_train, autoencoder_epochs)
        supervisor_losses = self.fit_supervisor(x_train, supervisor_epochs)
        dynamic_losses = self.fit_dynamic_game(x_train, dynamic_epochs, k)

        return autoencoder_losses, supervisor_losses, dynamic_losses


    ####################################################################################################################
    # Helper Functions
    ####################################################################################################################

    # returns num_samples of random normal noise in correct input shape for generator
    def get_noise(self, num_samples):

        return tf.random.normal((num_samples, self.dimensions.get(seq_length), self.dimensions.get(num_features)))

    # gets one instance of a batch from the data
    def batch_data(self, x_train):

        dataset = tf.data.Dataset.from_tensor_slices(x_train)
        dataset = dataset.shuffle(buffer_size=len(x_train))
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.repeat()
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        return dataset





            


