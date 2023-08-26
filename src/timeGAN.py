import numpy as np
import os
import datetime
from tensorflow.keras.models import Model
import tensorflow as tf
from src import networkparts

"""
This module defines the TimeGAN model. 

Inputs:
    - model_dimensions: a dict containing all of the dimension needed for the model, namely
        - seq_length: length of time steps
        - input_features: number of features given to the embedder
        - output_features: number of features to output
        - embedded_dims: number of dimensions desired for hidden layers
    - model_parameters: a dict containing all of the parameters used to set up the model
        - n_layers: number of layers desired for each time step
        - mu: regularizes the generator supervised loss when training the generator
        - phi: regularizes the embedder supervised loss when training the supervisor
        - lambda: regularizes the supervised loss when training the embedder
        - alpha_1: learning rate for the Adam optimizer
        - alpha_2: learning rate for the Adam optimizer
        - theta: scaling factor for the tanh function
        - is_volatility: determines if supervised loss is based on the regular values or volatility (squared values)
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
        input_features = model_dimensions.get("input_features")
        output_features = model_dimensions.get("output_features")
        embed = model_dimensions.get("embedded_dims")
        n = model_parameters.get("n_layers")
        theta = model_parameters.get("theta")

        # then add the network parts
        self.generator = networkparts.Generator((seq_len, input_features), embed, n, theta).build_network_part()
        self.discriminator = networkparts.Discriminator((seq_len, embed), n).build_network_part()
        self.supervisor = networkparts.Supervisor((seq_len, embed), embed, n, theta).build_network_part()
        self.embedder = networkparts.Embedder((seq_len, input_features), embed, n, theta).build_network_part()
        self.recovery = networkparts.Recovery((seq_len, embed), output_features, n, theta).build_network_part()

    # compiles the model before taining the model
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

    # saves all of the network parts to a specific location, taking in a base location to store everything,
    # and directory name is either the current datetime or optional name argument
    def save_models(self, input_dir, name=None):

        # Create the directory based on optional name argument
        if name is None:
            # Create a subdirectory with "model" + "current datetime" as its name
            model_subdir = os.path.join(input_dir, "model_" + datetime.datetime.now().strftime("%Y%m%d_%H%M"))
        else:
            # Create a subdirectory with "model" + "name" as its name
            model_subdir = os.path.join(input_dir, "model_" + name)

        # Make the subdirectory
        os.makedirs(model_subdir)

        # Save the Generator model
        generator_path = os.path.join(model_subdir, "generator")
        self.generator.save(generator_path)

        # Save the Discriminator model
        discriminator_path = os.path.join(model_subdir, "discriminator")
        self.discriminator.save(discriminator_path)

        # Save the Supervisor model
        supervisor_path = os.path.join(model_subdir, "supervisor")
        self.supervisor.save(supervisor_path)

        # Save the Embedder model
        embedder_path = os.path.join(model_subdir, "embedder")
        self.embedder.save(embedder_path)

        # Save the Recovery model
        recovery_path = os.path.join(model_subdir, "recovery")
        self.recovery.save(recovery_path)

    # loads all of the network parts of the network model, taking in the path created above
    def load_models(self, base_directory, model_directory):

        # model part names
        generator_module = "generator"
        discriminator_module = "discriminator"
        supervisor_module = "supervisor"
        embedder_module = "embedder"
        recovery_module = "recovery"

        # Construct the full path to the model directory
        full_model_directory = os.path.join(base_directory, model_directory)

        # Load the Generator model
        generator_path = os.path.join(full_model_directory, generator_module)
        self.generator = tf.keras.models.load_model(generator_path)

        # Load the Discriminator model
        discriminator_path = os.path.join(full_model_directory, discriminator_module)
        self.discriminator = tf.keras.models.load_model(discriminator_path)

        # Load the Supervisor model
        supervisor_path = os.path.join(full_model_directory, supervisor_module)
        self.supervisor = tf.keras.models.load_model(supervisor_path)

        # Load the Embedder model
        embedder_path = os.path.join(full_model_directory, embedder_module)
        self.embedder = tf.keras.models.load_model(embedder_path)

        # Load the Recovery model
        recovery_path = os.path.join(full_model_directory, recovery_module)
        self.recovery = tf.keras.models.load_model(recovery_path)


    ####################################################################################################################
    # This part defines the gradient steps for each network part to be used later in full training
    ####################################################################################################################

    """
    This trains the embedder, recovery, and supervisor function simultaneously. The losses and network parts
    break down as follows:
    
    - Reconstruction: The embedder and recovery make up the two way mapping between the embedded and regular representation 
    of the data. They are trained to minimize the euclidian distance between the real data, and the data recovered from 
    the embedder. This ensures that both an effective embedded representation, as well as an effective way of recovering 
    the data from this embedded represent, is found (so samples from the generator can be recovered after training).
    
    - Supervised: At the same time the supervisor is trained to capture the stepwise dynamics of the real data in the embedded 
    space. The stepwise dynamics are learned by minimizing the distance between it's output and the output of the next 
    timestep in the real data. This way the supervisor learns how steps within the real data are taken.  By training the 
    supervisor in conjunction with the embedder, the embedder also learns a representation that's more favorable for the 
    supervisor. 
    
    """
    def train_autoencoder_step(self, batch):

        # define input
        X = batch

        # combine the trainable variables before training so they can be kept track of in tape
        trainable_vars = self.embedder.trainable_variables + \
                         self.recovery.trainable_variables + \
                         self.supervisor.trainable_variables


        with tf.GradientTape() as tape:

            # watch the trainable variables
            tape.watch(trainable_vars)

            # compute the recovered and supervised data
            E = self.embedder(X, training=True)
            X_hat = self.recovery(E, training=True)

            # compute the losses
            R_loss = self.reconstruction_loss(X, X_hat)

            # compute the supervised losses
            S_loss = None

            # this section deals with the
            if self.model_parameters.get("is_volatility"):

                # feed the supervisor the current volatilities
                H = self.supervisor(tf.square(E), training=True)

                # if supervised learning is type volatility then square the outputs to measure volatility
                S_loss = self.supervised_loss(tf.square(E[:, 1:, :]), tf.square(H[:, :-1, :]))

            else:

                # get the normal supervised output
                H = self.supervisor(E, training=True)

                # else keep the outputs the same
                S_loss = self.supervised_loss(E[:, 1:, :], H[:, :-1, :])

            # combine the losses
            total_loss = R_loss + self.model_parameters.get("lambda")*S_loss

        # compute and apply the gradient
        grads = tape.gradient(total_loss, trainable_vars)
        tf.keras.optimizers.Adam(learning_rate=self.model_parameters.get("alpha_1")).apply_gradients(
            zip(grads, trainable_vars))

        return S_loss, R_loss

    """
    This trains the generator in the dynamic game with the discriminator. It simultaneously updates the generator,
    embedder, supervisor, and recovery. The losses and network parts breaks down as follows:
    
    - Supervised_generator: The generator output is fed through the supervisor, and the new supervised loss is 
    computed with respect to the generator output. Since the supervisor is trained to mimic the stepwise dynamics of 
    the real data, this new loss forces the generators representation to adhere to those same stepwise dynamics by
    minimizing the distance between what the supervisor thinks the next timestep would be if the data was real,
    and the generators actual next timestep. 
    
    - Supervised_embedder: At the same time the supervisor is fed real data from the embedder, and computes the 
    supervised loss with respect to the real embedded output. By updating the supervisor on both supervised losses, it
    synchronize the embedded representation with the generators representation. By adapting to the generator loss, the 
    supervisor captures how the generator currently performs its step. Then the embedded loss ensures that the supervisor
    still adheres to the stepwise properties of the real data. At the same time, when the embedder also updated on the 
    supervised loss. Since the supervisor now also takes into account the generators stepwise properties, this forces
    the embedder to adapt its representation closer to that of the generator while remaining an accurate embedded
    representation of the real data. Overall, this joint minimization procedure aids the generator in learning the stepwise 
    properties of the real data.
    
    - Reconstruction: This is the same as the reconstruction in the autoencoder training, and simply ensures that both
    the embedder and the generator still create an effective latent mapping for the data.
    
    - Unsupervised: The generator output is fed to the discriminator, generating a label for whether or not the discriminator
    thinks this output is real. This output is then given the target label corresponding to real data (zero in this case),
    and the binary cross entropy is computed as the loss. By minimizing this loss, the generator is rewarded for getting
    the discriminator to label it's output as real.
    
    The loss functions are combined in two separate tapes, so the tuning parameters to be incorporated correctly.
    """
    def train_generator_step(self, batch):

        # define input
        Z = self.get_noise(self.batch_size)
        X = batch

        # combine first set of trainable variables before training so they can be kept track of in tape
        trainable_vars_1 = (self.generator.trainable_variables +
                          self.embedder.trainable_variables +
                          self.recovery.trainable_variables)

        # create the tape for the first loss function
        with tf.GradientTape() as tape_1:
            # watch the trainable variables
            tape_1.watch(trainable_vars_1)

            # compute the all of network outputs
            E = self.embedder(X, training=True)
            E_hat = self.generator(Z, training=True)
            X_hat = self.recovery(E, training=True)
            Y_hat = self.discriminator(E_hat, training=False)
            Y = tf.zeros_like(Y_hat)

            output_features = self.model_dimensions.get("output_features")

            # compute the losses
            R_loss = self.reconstruction_loss(X[:,:,:output_features], X_hat)
            U_loss = self.unsupervised_loss(Y, Y_hat)

            # supervised losses
            S_loss_e = None
            S_loss_g = None

            if self.model_parameters.get("is_volatility"):

                # get the supervised outputs
                H_e = self.supervisor(E, training=True)
                H_g = self.supervisor(E_hat, training=True)

                # if supervised learning is type volatility then square the outputs to measure volatility
                S_loss_e = self.supervised_loss(tf.square(E[:, 1:, :]), tf.square(H_e[:, :-1, :]))
                S_loss_g = self.supervised_loss(tf.square(E_hat[:, 1:, :]), tf.square(H_g[:, :-1, :]))

            else:

                # get the supervised outputs
                H_e = self.supervisor(E, training=True)
                H_g = self.supervisor(E_hat, training=True)

                # else keep the outputs the same
                S_loss_e = self.supervised_loss(E[:, 1:, :], H_e[:, :-1, :])
                S_loss_g = self.supervised_loss(E_hat[:, 1:, :], H_g[:, :-1, :])

            # combine the losses
            total_loss_1 = U_loss + \
                         R_loss + \
                         self.model_parameters.get("lambda")*S_loss_e + \
                         self.model_parameters.get("mu")*S_loss_g


        # define the second set of variables (ie just the supervisor)
        trainable_vars_2 = self.supervisor.trainable_variables

        # create the tape for the second loss function
        with tf.GradientTape() as tape_2:
            # watch the trainable variables
            tape_2.watch(trainable_vars_2)

            # compute the all of network outputs
            E = self.embedder(X, training=True)
            E_hat = self.generator(Z, training=True)
            H_e = self.supervisor(E, training=True)
            H_g = self.supervisor(E_hat, training=True)

            # compute the losses
            S_loss_e = None
            S_loss_g = None

            if self.model_parameters.get("is_volatility"):
                # if supervised learning is type volatility then square the outputs to measure volatility
                S_loss_e = self.supervised_loss(tf.square(E[:, 1:, :]), tf.square(H_e[:, :-1, :]))
                S_loss_g = self.supervised_loss(tf.square(E_hat[:, 1:, :]), tf.square(H_g[:, :-1, :]))

            else:
                # else keep the outputs the same
                S_loss_e = self.supervised_loss(E[:, 1:, :], H_e[:, :-1, :])
                S_loss_g = self.supervised_loss(E_hat[:, 1:, :], H_g[:, :-1, :])

            # combine the losses
            total_loss_2 = S_loss_e + \
                         self.model_parameters.get("phi") * S_loss_g

        # compute and apply the gradient for the two groups
        grads_1 = tape_1.gradient(total_loss_1, trainable_vars_1)
        grads_2 = tape_2.gradient(total_loss_2, trainable_vars_2)
        tf.keras.optimizers.Adam(learning_rate=self.model_parameters.get("alpha_2")).apply_gradients(
            zip(grads_1, trainable_vars_1))
        tf.keras.optimizers.Adam(learning_rate=self.model_parameters.get("alpha_2")).apply_gradients(
            zip(grads_2, trainable_vars_2))

        return S_loss_e, S_loss_g, R_loss, U_loss


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
            Y_hat = tf.concat([Y_hat_real, Y_hat_fake], axis=0)

            # Create the real labels for the discriminator outputs
            Y = tf.concat([tf.zeros_like(Y_hat_real), tf.ones_like(Y_hat_fake)], axis=0)

            # compute the unsupervised discriminator loss
            discriminator_loss = self.unsupervised_loss(Y, Y_hat)


        # compute and apply the gradient
        trainable_variables = self.discriminator.trainable_variables
        grad = tape.gradient(discriminator_loss, trainable_variables)
        tf.keras.optimizers.Adam(learning_rate=self.model_parameters.get("alpha_2")).apply_gradients(
            zip(grad, trainable_variables))

        return discriminator_loss


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

        # create empty array to store the loss at each step of each epoch
        losses = []

        # iterate over the data epochs number of times
        for epoch in range(epochs):

            # generate the batched data set
            batched_data = self.batch_data(x_train)

            # create an empty array for the loss at each step in epoch
            epoch_losses = []

            # iterate over all batches in the batched data set
            for step, batch in enumerate(batched_data):

                # train the model and return the reconstruction loss
                S_loss, R_loss = self.train_autoencoder_step(batch)

                # store current losses
                epoch_losses.append([S_loss, R_loss])

                print(f"Epoch {epoch}, step {step}: Reconstruction loss = {R_loss}, Supervised loss = {S_loss}")

            # append the average losses for the epoch
            losses.append([np.mean(np.array(epoch_losses)[:,0]), np.mean(np.array(epoch_losses)[:,1])])

        print(f"Finished Autoencoder Training")

        return {"Supervised Loss": [loss[0] for loss in losses], "Reconstruction Loss": [loss[1] for loss in losses]}

    """
    This runs the full training for the dynamic game. This is a little more involved. First the generator is trained 
    for k steps, to allow for stronger data generation, then the discriminator is trained for 1 step. And this is done 
    for each epoch.
    """
    def fit_dynamic_game(self, x_train, epochs, k):

        print(f"Starting Dynamic Training")

        # create empty array to store the loss at each step of each epoch
        losses = []

        # iterate over the data autoencoder_epochs number of times
        for epoch in range(epochs):

            # generate the batched data set
            batched_data = self.batch_data(x_train)

            # create an empty array for the loss at each step in epoch
            epoch_losses = []

            # iterate over all batches in the batched data set
            for step, batch in enumerate(batched_data):

                # in every step run the generator sequence
                S_loss_e, S_loss_g, R_loss, U_loss_g = self.train_generator_step(batch)


                # train the discriminator every k steps
                U_loss_d = None
                if step % k == k-1:
                    U_loss_d = self.train_discriminator_step(batch)


                # append the three losses to the epoch losses
                epoch_losses.append([R_loss, S_loss_e, S_loss_g, U_loss_g, U_loss_d])

                print(f"Epoch {epoch}, step {step}: "
                      f"Reconstruction loss = {R_loss}, "
                      f"Supervised E loss = {S_loss_e}, "
                      f"Supervised G loss = {S_loss_g}, "
                      f"Unsupervised G loss = {U_loss_g}, "
                      f"Unsupervised D loss = {U_loss_d}")

            # Next compute the average of each loss from the epoch and append it
            uld = np.mean(np.array([row[4] for row in epoch_losses if row[4] is not None]))

            epoch_losses = np.array(epoch_losses, dtype=np.float32)
            rl = np.mean(epoch_losses[:, 0])
            sle = np.mean(epoch_losses[:, 1])
            slg = np.mean(epoch_losses[:, 2])
            ulg = np.mean(epoch_losses[:, 3])


            losses.append([rl, sle, slg, ulg, uld])

        print(f"Finished Dynamic Training")

        return {"Reconstruction Loss": [loss[0] for loss in losses],
                "Supervised Embedder Loss": [loss[1] for loss in losses],
                "Supervised Generator Loss": [loss[2] for loss in losses],
                "Unsupervised Generator Loss": [loss[3] for loss in losses],
                "Unsupervised Discriminator Loss": [loss[4] for loss in losses]}


    ####################################################################################################################
    # Helper Functions
    ####################################################################################################################

    # returns num_samples of random normal noise in correct input shape for generator
    def get_noise(self, num_samples):

        return tf.random.normal((num_samples, self.model_dimensions.get("seq_length"), self.model_dimensions.get("input_features")))

    # gets one instance of a batch from the data
    def batch_data(self, x_train):

        dataset = tf.data.Dataset.from_tensor_slices(x_train)
        dataset = dataset.shuffle(buffer_size=len(x_train))
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        return dataset

    def generate_seq(self, samples):

        Z = self.get_noise(samples)
        E_hat = self.generator(Z)
        X_hat = self.recovery(E_hat)

        return tf.squeeze(X_hat)

    def autoencode_seq(self, sequences):

        # if a single sequence is passed in expand the dimension
        seq = np.expand_dims(sequences, axis=0)

        # if more than one sequence is passed in, randomly pick one of them
        if sequences.ndim == 3:
            random_index = np.random.randint(len(sequences))
            seq = np.expand_dims(sequences[random_index, :, :], axis=0)

        # use the sequence to get the recovered sequence
        X = seq
        E = self.embedder(X)
        X_hat = self.recovery(E)

        return tf.squeeze(X_hat)









            


