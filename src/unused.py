# file to park all unused code

"""
    ####################################################################################################################
    # Below are the training functions for each individual part of the network and the full fit functions.
    # Running the fit for individual parts of the network allows the user to optimize the training for each part
    # of the network separately, without having to train the full network.
    ####################################################################################################################

    # This runs the full training for the autoencoder
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
                reconstruction_loss = self.train_autoencoder_step(batch)

                # append the loss
                epoch_losses.append(reconstruction_loss)

                print(f"Epoch {epoch}, step {step}: Reconstruction loss = {reconstruction_loss}")

            losses.append(epoch_losses)

        print(f"Finished Autoencoder Training")

        return losses


    # This runs the full training for the supervisor

    def fit_supervisor(self, x_train, epochs):

        print(f"Starting Supervisor Training")

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
                supervised_loss = self.train_supervisor_step(batch)

                # append the loss
                epoch_losses.append(supervised_loss)

                print(f"Epoch {epoch}, step {step}: Supervised loss = {supervised_loss}")

            losses.append(epoch_losses)

        print(f"Finished Supervisor Training")

        return losses


    #This runs the full training for the dynamic game. This is a little more involved. First the generator is trained
    #for k steps, to allow for stronger data generation, then the discriminator is trained for 1 step. And this is done
    #for each epoch.

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

                # we then train the generator k times at each step

                # first create a temporary holder for the generator losses
                gen_s_loss = 0
                gen_u_loss = 0

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



    #Finally define a fit function for training the whole TimeGAN

    def fit(self, x_train, autoencoder_epochs, supervisor_epochs, dynamic_epochs, k):

        autoencoder_losses = self.fit_autoencoder(x_train, autoencoder_epochs)
        supervisor_losses = self.fit_supervisor(x_train, supervisor_epochs)
        dynamic_losses = self.fit_dynamic_game(x_train, dynamic_epochs, k)

        return autoencoder_losses, supervisor_losses, dynamic_losses

"""