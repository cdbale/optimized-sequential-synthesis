"""
# run below to get started
"""

import math
import numpy as np
import statistics
from sklearn import metrics
from __future__ import print_function, division
from functools import partial
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import argparse
import keras
from tensorflow.keras import backend as K
from google.colab import drive
from google.colab import files
from sklearn.linear_model import LinearRegression
import sys
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
import pandas as pd
import io
from keras.models import load_model
import time
from scipy.stats import pearsonr
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, GaussianNoise
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D
from keras.layers import MaxPooling2D, LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D, Conv1D
from keras.models import Sequential, Model
from keras import losses
import keras.backend as K
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity
import os
from sklearn.model_selection import train_test_split
import random

# set global seeds
seed=1
os.environ['PYTHONHASHSEED'] = str(seed)
# For working on GPUs from "TensorFlow Determinism"
os.environ["TF_DETERMINISTIC_OPS"] = str(seed)
np.random.seed(seed)
random.seed(seed)
tf.random.set_seed(seed)
print(random.random())

# # define utility
# def utility(real_data, protected_data):
#   from sklearn.linear_model import LinearRegression
#   from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error
#   reg = LinearRegression()
#   reg.fit(np.array(real_data)[:,1:9],np.array(real_data)[:,0])
#   reg_protect = LinearRegression()
#   reg_protect.fit(np.array(protected_data)[:,1:9],np.array(protected_data)[:,0])
#   MAPD = mean_absolute_percentage_error(reg.coef_, reg_protect.coef_)*100
#   MAE = mean_absolute_error(reg.coef_, reg_protect.coef_)
#   MSE = mean_squared_error(reg.coef_, reg_protect.coef_)
#   return MAPD, MAE, MSE

"""# Anand and lee (2022)"""

## in the paper we had the following optimal settings:
N = 1262423
samples = N
iterations = (100000)+1
batch_size = 128

epochs = iterations/(N/batch_size)
print(epochs)

# therefore we want to have the same number of epochs for smaller sample sizes
N = 10000
samples = int(N*3)
iterations = 1000
batch_size = 100
epochs = iterations/(N/batch_size)
print(epochs)

class GAN():
    def __init__(self, privacy):
      self.img_rows = 1
      self.img_cols = 1
      self.img_shape = (self.img_cols,)
      self.latent_dim = (1)

      optimizer = keras.optimizers.Adam()
      self.discriminator = self.build_discriminator()
      self.discriminator.compile(loss='binary_crossentropy',
                                 optimizer=optimizer,
                                 metrics=['accuracy'])
      if privacy == True:
        print("using differential privacy")
        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(optimizer=DPKerasAdamOptimizer(
            l2_norm_clip=l2_norm_clip,
            noise_multiplier=noise_multiplier,
            num_microbatches=num_microbatches,
            learning_rate=lr),
            loss= tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.losses.Reduction.NONE), metrics=['accuracy'])

      # Build the generator
      self.generator = self.build_generator()

      # The generator takes noise as input and generates imgs
      z = Input(shape=(self.latent_dim,))
      img = self.generator(z)

      # For the combined model we will only train the generator
      self.discriminator.trainable = False

      # The discriminator takes generated images as input and determines validity
      valid = self.discriminator(img)

      # The combined model  (stacked generator and discriminator)
      # Trains the generator to fool the discriminator
      self.combined = Model(z, valid)
      self.combined.compile(loss='binary_crossentropy', optimizer= optimizer)


    def build_generator(self):
      model = Sequential()
      model.add(Dense(self.latent_dim, input_dim=self.latent_dim))
      model.add(LeakyReLU(alpha=0.2))
      #model.add(BatchNormalization())
      model.add(Dense(1024, input_shape=self.img_shape))
      model.add(LeakyReLU(alpha=0.2))
      #model.add(BatchNormalization())
      model.add(Dense(self.latent_dim))
      model.add(Activation("tanh"))

      #model.summary()

      noise = Input(shape=(self.latent_dim,))
      img = model(noise)
      return Model(noise, img)

    def build_discriminator(self):

        model = Sequential()

        model.add(Dense(1024, input_shape=self.img_shape))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1, activation='sigmoid'))

        #model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)

    def train(self, data, iterations, batch_size, model_name, generator_losses = [], discriminator_acc = [], correlations = [], accuracy = [], MAPD_col = [],MSE_col = [], MAE_col = []):
      # Adversarial ground truths

      valid = np.ones((batch_size, 1))
      fake = np.zeros((batch_size, 1))
      corr = 0
      MAPD = 0
      MSE = 0
      MAE = 0
      #fake += 0.05 * np.random.random(fake.shape)
      #valid += 0.05 * np.random.random(valid.shape)

      for epoch in range(iterations):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random batch of images
            idx = np.random.randint(0, data.shape[0], batch_size)
            imgs = data[idx]

            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

            # Generate a batch of new images
            gen_imgs = self.generator.predict(noise, verbose = False)

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------
            # Train the generator (to have the discriminator label samples as valid)

            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            g_loss = self.combined.train_on_batch(noise, valid)

            # collect losses
            discriminator_acc = np.append(discriminator_acc, 100*d_loss[1])
            generator_losses = np.append(generator_losses, g_loss)
      self.generator.save(model_name)
              #print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f, corr: %f, MAPD: %f, MSE: %f, MAE: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss, corr, MAPD, MSE, MAE))

random.seed(1)
np.random.seed(1)
tf.random.set_seed(1)
epsilons = np.array([])
MAPD_col = np.array([])
MAE_col = np.array([])
MSE_col = np.array([])

for iter in range(0,100):
  random.seed(iter)
  np.random.seed(iter)
  tf.random.set_seed(iter)
  churn = pd.read_csv('data.csv', sep = ',', na_values=['(NA)']).fillna(0)
  churn = pd.DataFrame.drop_duplicates(churn)
  churn, evaluation_outside_training = train_test_split(churn, train_size = int(samples*2/3), test_size = int(30000), stratify= churn['Churn'])
  train_original, adversary_training = train_test_split(churn, train_size = int(samples*1/3), stratify= churn['Churn'])
  N = len(train_original)/10

  train_outcome = train_original[['Tenure']]
  train_covariates = train_original.drop('Tenure', axis=1)

  adversary_training_outcome = adversary_training[['Tenure']]
  adversary_training_covariates = adversary_training.drop('Tenure', axis=1)

  from sklearn.preprocessing import MinMaxScaler
  scaler0 = MinMaxScaler(feature_range= (-1, 1))
  scaler0 = scaler0.fit(train_outcome)
  train_outcome = scaler0.transform(train_outcome)
  train_outcome = pd.DataFrame(train_outcome)

  print("start train set training")
  gan_train = GAN(privacy = False)
  gan_train.train(data = np.array(train_outcome), iterations=iterations, batch_size=batch_size, model_name = "train_anand.h5")

  # Generate a batch of new customers
  generator = load_model('train_anand.h5', compile = True)
  noise = np.random.normal(0, 1, (len(train_outcome), 1))
  gen_imgs = generator.predict(noise, verbose = False)
  gen_imgs = scaler0.inverse_transform(gen_imgs)
  gen_imgs = gen_imgs.reshape(len(train_outcome), 1)
  train_GAN = pd.DataFrame(gen_imgs)

  # adversary has access to the model and samples another adversary_sample
  print("start adversary set training")

  from sklearn.preprocessing import MinMaxScaler
  scaler1 = MinMaxScaler(feature_range= (-1, 1))
  scaler1 = scaler1.fit(adversary_training_outcome)
  adversary_training_outcome = scaler1.transform(adversary_training_outcome)
  adversary_training_outcome = pd.DataFrame(adversary_training_outcome)

  gan_adv = GAN(privacy = False)
  gan_adv.train(data = np.array(adversary_training_outcome), iterations=iterations, batch_size=batch_size, model_name = "adversary_anand.h5")

  generator = load_model('adversary_anand.h5', compile = True)
  generated_data = []

  noise = np.random.normal(0, 1, (len(adversary_training_outcome), 1))
  # Generate a batch of new images
  gen_imgs = generator.predict(noise, verbose = False)
  gen_imgs = scaler1.inverse_transform(gen_imgs)
  gen_imgs = gen_imgs.reshape(len(adversary_training_outcome), 1)
  adversary_training_GAN = pd.DataFrame(gen_imgs)

  # combine one protected variable with other
  train = pd.concat([train_covariates.reset_index(drop = True), train_GAN], axis=1)
  adversary = pd.concat([adversary_training_covariates.reset_index(drop = True), adversary_training_GAN], axis=1)

  # stap 1, 2
  train.rename(columns = {0:'Tenure'}, inplace = True)
  adversary.rename(columns = {0:'Tenure'}, inplace = True)
  params = {"bandwidth": np.logspace(-1, 1, 20)}
  grid_train = GridSearchCV(KernelDensity(), params, n_jobs = -1)
  grid_train.fit(train)
  kde_train = grid_train.best_estimator_

  params = {"bandwidth": np.logspace(-1, 1, 20)}
  grid = GridSearchCV(KernelDensity(), params, n_jobs = -1)
  grid.fit(adversary)
  kde_adversary = grid.best_estimator_
  evaluation_outside_training = evaluation_outside_training[['Churn','Sex', 'Age', 'Contact', 'Household_size', 'Social_class', 'Income', 'Ethnicity', 'Tenure']]

  # stap 3
  density_train = kde_train.score_samples(train) # f1
  density_adversary = kde_adversary.score_samples(train) # f2
  #print(density_train > density_adversary)  # f1 > f2
  TPR = sum(density_train > density_adversary)/len(density_train) # all training!

  # stap 4
  density_train_new = kde_train.score_samples(evaluation_outside_training) # f1
  density_adversary_new = kde_adversary.score_samples(evaluation_outside_training) # f2
  #density_train_new > density_adversary_new  # f1 > f2
  #print(density_train_new > density_adversary_new)  # f1 > f2
  FPR = sum(density_train_new > density_adversary_new)/len(density_train_new) # random!
  TNR = 1 - FPR
  FNR = 1 - TPR
  print("FPR is " + str(FPR))
  print("FNR is " + str(FNR))
  print("TPR is " + str(TPR))
  print("TNR is " + str(TNR))
  try:
    epsilons = np.append(epsilons,max(math.log((1 - (1/N) - FPR)/FNR), math.log((1 - (1/N) - FNR)/FPR)))
    print("empirical epsilon = " + str(max(math.log((1 - (1/N) - FPR)/FNR), math.log((1 - (1/N) - FNR)/FPR))))
  except:
    epsilons = np.append(epsilons, math.log((1 - (1/N) - FPR)/FNR))
    print("empirical epsilon = " + str(math.log((1 - (1/N) - FPR)/FNR)))

  # utility
  MAPD_train, MAE_train, MSE_train = utility(real_data = train, protected_data = train_GAN)
  MAPD_adv, MAE_adv, MSE_adv = utility(real_data = train, protected_data = adversary_training_GAN)
  MAPD_col = np.append(MAPD_col, ((MAPD_train+MAPD_adv)/2))
  MAE_col = np.append(MAE_col, ((MAE_train+MAE_adv)/2))
  MSE_col = np.append(MSE_col, ((MSE_train+MSE_adv)/2))
  print("MAPD train = " + str(MAPD_train))
  print("MAPD adversary = " + str(MAPD_adv))

np.savetxt("epsilons_anand_30000.csv", epsilons, delimiter=",")
np.savetxt("MAPD_anand_30000.csv", MAPD_col, delimiter=",")
np.savetxt("MAE_anand_30000.csv", MAE_col, delimiter=",")
np.savetxt("MSE_anand_30000.csv", MSE_col, delimiter=",")

"""# GANs with differential privacy"""

!pip install tensorflow_privacy --quiet

class GAN():
    def __init__(self, privacy):
      self.img_rows = 1
      self.img_cols = 9
      self.img_shape = (self.img_cols,)
      self.latent_dim = (9)
      lr = 0.001

      optimizer = keras.optimizers.Adam()
      self.discriminator = self.build_discriminator()
      self.discriminator.compile(loss='binary_crossentropy',
                                 optimizer=optimizer,
                                 metrics=['accuracy'])
      if privacy == True:
        print(noise_multiplier)
        print("using differential privacy")
        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(optimizer=DPKerasAdamOptimizer(
            l2_norm_clip=4,
            noise_multiplier=noise_multiplier,
            num_microbatches=num_microbatches,
            learning_rate=lr),
            loss= tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.losses.Reduction.NONE), metrics=['accuracy'])

      # Build the generator
      self.generator = self.build_generator()

      # The generator takes noise as input and generates imgs
      z = Input(shape=(self.latent_dim,))
      img = self.generator(z)

      # For the combined model we will only train the generator
      self.discriminator.trainable = False

      # The discriminator takes generated images as input and determines validity
      valid = self.discriminator(img)

      # The combined model  (stacked generator and discriminator)
      # Trains the generator to fool the discriminator
      self.combined = Model(z, valid)
      self.combined.compile(loss='binary_crossentropy', optimizer= optimizer)


    def build_generator(self):
      model = Sequential()
      model.add(Dense(self.latent_dim, input_dim=self.latent_dim))
      model.add(LeakyReLU(alpha=0.2))
      #model.add(BatchNormalization())
      model.add(Dense(64, input_shape=self.img_shape))
      model.add(LeakyReLU(alpha=0.2))
      #model.add(BatchNormalization())
      model.add(Dense(self.latent_dim))
      model.add(Activation("tanh"))

      #model.summary()

      noise = Input(shape=(self.latent_dim,))
      img = model(noise)
      return Model(noise, img)

    def build_discriminator(self):

        model = Sequential()

        model.add(Dense(64, input_shape=self.img_shape))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1, activation='sigmoid'))

        #model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)

    def train(self, data, iterations, batch_size, sample_interval, model_name, generator_losses = [], discriminator_acc = [], correlations = [], accuracy = [], MAPD_collect = [],MSE_collect = [], MAE_collect = []):
      # Adversarial ground truths
      valid = np.ones((batch_size, 1))
      fake = np.zeros((batch_size, 1))
      corr = 0
      MAPD = 0
      MSE = 0
      MAE = 0
      #fake += 0.05 * np.random.random(fake.shape)
      #valid += 0.05 * np.random.random(valid.shape)

      for epoch in range(iterations):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random batch of images
            idx = np.random.randint(0, data.shape[0], batch_size)
            imgs = data[idx]

            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

            # Generate a batch of new images
            gen_imgs = self.generator.predict(noise, verbose = False)

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------
            # Train the generator (to have the discriminator label samples as valid)

            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            g_loss = self.combined.train_on_batch(noise, valid)

            if (epoch % 100) == 0:
              print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

      print("save model")
      self.generator.save(model_name)

"""iteraties en batch size hetzelfde houden."""

from keras.models import load_model
from absl import app
from absl import flags
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import logging
from tensorflow_privacy.privacy.analysis import compute_dp_sgd_privacy
from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras import DPKerasSGDOptimizer, DPKerasAdamOptimizer
from tensorflow_privacy.privacy.analysis.compute_dp_sgd_privacy_lib import compute_dp_sgd_privacy
from sklearn.preprocessing import MinMaxScaler

random.seed(1)
np.random.seed(1)
tf.random.set_seed(1)

start_time = time.time()

samples = 300
output_samples = 300 # we always set this to 3000, except with n = 30,000, then 30,000
churn = pd.read_csv('data.csv', sep = ',', na_values=['(NA)']).fillna(0)
churn = pd.DataFrame.drop_duplicates(churn)
print(samples)

epsilons_13 = np.array([])
MAPD_col_13 = np.array([])
MAE_col_13 = np.array([])
MSE_col_13 = np.array([])

epsilons_3 = np.array([])
MAPD_col_3 = np.array([])
MAE_col_3 = np.array([])
MSE_col_3 = np.array([])

epsilons_1 = np.array([])
MAPD_col_1 = np.array([])
MAE_col_1 = np.array([])
MSE_col_1 = np.array([])

epsilons_05 = np.array([])
MAPD_col_05 = np.array([])
MAE_col_05 = np.array([])
MSE_col_05 = np.array([])

epsilons_005 = np.array([])
MAPD_col_005 = np.array([])
MAE_col_005 = np.array([])
MSE_col_005 = np.array([])

epsilons_001 = np.array([])
MAPD_col_001 = np.array([])
MAE_col_001 = np.array([])
MSE_col_001 = np.array([])

TPR_col = np.array([])
FPR_col = np.array([])
TNR_col = np.array([])
FNR_col = np.array([])

# for 300 obs, 10 iteraties, 100 batch size, epochs = 10
noise_multipliers = [1.011, 2.98, 6.96, 11.85, 60]

# for 3,000 obs, 100 iteraties, 100 batch size, epochs = 10
#noise_multipliers = [0.6785, 1.43, 3.1, 5.4, 38.5]

# for 30,000 obs, 1000 iteraties, 100 batch size, epochs = 10.
#noise_multipliers = [0.502, 0.81, 1.35, 2.23, 15.5]

for iter in range(100):
  random.seed(iter)
  np.random.seed(iter)
  tf.random.set_seed(iter)
  print("iteration is " + str(iter))
  churn = pd.read_csv('data.csv', sep = ',', na_values=['(NA)']).fillna(0).sample(frac = 1)
  churn = pd.DataFrame.drop_duplicates(churn)
  churn, evaluation_outside_training = train_test_split(churn, train_size = int(samples*2/3), test_size = int(30000), stratify= churn['Churn'])
  train, adversary_training = train_test_split(churn, train_size = int(samples*1/3), stratify= churn['Churn'])

  scaler0 = MinMaxScaler(feature_range= (-1, 1))
  scaler0 = scaler0.fit(train)
  train_GAN = scaler0.transform(train)
  train_GAN = pd.DataFrame(train_GAN)

  scaler1 = MinMaxScaler(feature_range= (-1, 1))
  scaler1 = scaler1.fit(adversary_training)
  adversary_training_GAN = scaler1.transform(adversary_training)
  adversary_training_GAN = pd.DataFrame(adversary_training_GAN)

  for noise in noise_multipliers: # we vary the noise multipliers here
    random.seed(iter)
    np.random.seed(iter)
    tf.random.set_seed(iter)

    # setting epsilon
    N = len(train)
    batch_size = 100
    iterations = 10
    epochs = iterations/(N/batch_size) # should be 10

    noise_multiplier = noise
    l2_norm_clip = 4 # see paper in validation section.
    delta= 1/N # should be 1/N
    theor_epsilon =compute_dp_sgd_privacy(N, batch_size, noise_multiplier,
                          epochs, delta) # calculate the theoretical bound of epsilon
    N = len(train)/10 # to prevent naive model
    num_microbatches = batch_size # see validation section paper.
    print("theoretical epsilon = " + str(round(theor_epsilon[0],2))) # print epsilon

    # train GAN on train data
    gan_train = GAN(privacy = True)
    gan_train.train(data = np.array(train_GAN), iterations=iterations, batch_size=batch_size, sample_interval=((iterations-1)/10), model_name = "train_1.h5")

    # Generate a batch of new customers
    generator = load_model('train_1.h5')
    noise = np.random.normal(0, 1, (output_samples, 9))
    gen_imgs = generator.predict(noise, verbose = False)
    gen_imgs = scaler0.inverse_transform(gen_imgs)
    train_GAN = pd.DataFrame(gen_imgs.reshape(output_samples, 9))
    train_GAN.columns = train.columns.values

    # adversary has access to the model and samples another adversary_sample
    gan_adv = GAN(privacy = True)
    gan_adv.train(data = np.array(adversary_training_GAN), iterations=iterations, batch_size=batch_size, sample_interval=((iterations-1)/10), model_name = "adversary_1.h5")

    # Generate a batch of new images
    generator = load_model('adversary_1.h5')
    noise = np.random.normal(0, 1, (output_samples, 9))
    gen_imgs = generator.predict(noise, verbose = False)
    gen_imgs = scaler1.inverse_transform(gen_imgs)
    adversary_training_GAN = pd.DataFrame(gen_imgs.reshape(output_samples, 9))
    adversary_training_GAN.columns = adversary_training.columns.values

    # stap 1, 2
    params = {"bandwidth": np.logspace(-1, 1, 20)}
    grid_train = GridSearchCV(KernelDensity(), params, n_jobs = -1)
    grid_train.fit(train_GAN)
    print(grid_train.best_estimator_)
    kde_train = grid_train.best_estimator_

    grid = GridSearchCV(KernelDensity(), params, n_jobs = -1)
    grid.fit(adversary_training_GAN)
    print(grid.best_estimator_)
    kde_adversary = grid.best_estimator_

    # stap 3
    density_train = kde_train.score_samples(train)
    density_adversary = kde_adversary.score_samples(train)
    TPR = sum(density_train > density_adversary)/len(density_train)

    # stap 4
    density_train_new = kde_train.score_samples(evaluation_outside_training)
    density_adversary_new = kde_adversary.score_samples(evaluation_outside_training)
    FPR = sum(density_train_new > density_adversary_new)/len(density_train_new)
    TNR = 1 - FPR
    FNR = 1 - TPR
    print("FPR is " + str(FPR))
    print("FNR is " + str(FNR))
    print("TPR is " + str(TPR))
    print("TNR is " + str(TNR))

    TPR_col = np.append(TPR_col, TPR)
    FPR_col = np.append(FPR_col, FPR)
    TNR_col = np.append(TNR_col, TNR)
    FNR_col = np.append(FNR_col, FNR)

    # utility
    MAPD_train, MAE_train, MSE_train = utility(real_data = train, protected_data = train_GAN)
    MAPD_adv, MAE_adv, MSE_adv = utility(real_data = train, protected_data = adversary_training_GAN)
    MAPD = (MAPD_train+MAPD_adv)/2
    MAE = (MAE_train+MAE_adv)/2
    MSE = (MSE_train+MSE_adv)/2
    print("MAPD" + str(MAPD))

    ## to save the results per epsilon (a bit lazy admittedly).
    if noise_multiplier == noise_multipliers[0]:
      try:
        epsilons_13 = np.append(epsilons_13,max(math.log((1 - (1/N) - FPR)/FNR), math.log((1 - (1/N) - FNR)/FPR)))
        print("empirical epsilon = " + str(max(math.log((1 - (1/N) - FPR)/FNR), math.log((1 - (1/N) - FNR)/FPR))))
        MAPD_col_13 = np.append(MAPD_col_13, MAPD)
        MAE_col_13 = np.append(MAE_col_13, MAE)
        MSE_col_13 = np.append(MSE_col_13, MSE)
      except:
        print("undefined privacy risk")
        epsilons_13 = np.append(epsilons_13, 0)
        print("empirical epsilon = " + str(0))
        MAPD_col_13 = np.append(MAPD_col_13, MAPD)
        MAE_col_13 = np.append(MAE_col_13, MAE)
        MSE_col_13 = np.append(MSE_col_13, MSE)

    if noise_multiplier == noise_multipliers[1]:
      try:
        epsilons_3 = np.append(epsilons_3,max(math.log((1 - (1/N) - FPR)/FNR), math.log((1 - (1/N) - FNR)/FPR)))
        print("empirical epsilon = " + str(max(math.log((1 - (1/N) - FPR)/FNR), math.log((1 - (1/N) - FNR)/FPR))))
        MAPD_col_3 = np.append(MAPD_col_3, MAPD)
        MAE_col_3 = np.append(MAE_col_3, MAE)
        MSE_col_3 = np.append(MSE_col_3, MSE)
      except:
        print("undefined privacy risk")
        epsilons_3 = np.append(epsilons_3, 0)
        print("empirical epsilon = " + str(0))
        MAPD_col_3 = np.append(MAPD_col_3, MAPD)
        MAE_col_3 = np.append(MAE_col_3, MAE)
        MSE_col_3 = np.append(MSE_col_3, MSE)

    if noise_multiplier == noise_multipliers[2]:
      try:
        epsilons_1 = np.append(epsilons_1,max(math.log((1 - (1/N) - FPR)/FNR), math.log((1 - (1/N) - FNR)/FPR)))
        print("empirical epsilon = " + str(max(math.log((1 - (1/N) - FPR)/FNR), math.log((1 - (1/N) - FNR)/FPR))))
        MAPD_col_1 = np.append(MAPD_col_1, MAPD)
        MAE_col_1 = np.append(MAE_col_1, MAE)
        MSE_col_1 = np.append(MSE_col_1, MSE)
      except:
        print("undefined privacy risk")
        epsilons_1 = np.append(epsilons_1, 0)
        print("empirical epsilon = " + str(0))
        MAPD_col_1 = np.append(MAPD_col_1, MAPD)
        MAE_col_1 = np.append(MAE_col_1, MAE)
        MSE_col_1 = np.append(MSE_col_1, MSE)

    if noise_multiplier == noise_multipliers[3]:
      try:
        epsilons_05 = np.append(epsilons_05,max(math.log((1 - (1/N) - FPR)/FNR), math.log((1 - (1/N) - FNR)/FPR)))
        print("empirical epsilon = " + str(max(math.log((1 - (1/N) - FPR)/FNR), math.log((1 - (1/N) - FNR)/FPR))))
        MAPD_col_05 = np.append(MAPD_col_05, MAPD)
        MAE_col_05 = np.append(MAE_col_05, MAE)
        MSE_col_05 = np.append(MSE_col_05, MSE)
      except:
        print("undefined privacy risk")
        epsilons_05 = np.append(epsilons_05, 0)
        print("empirical epsilon = " + str(0))
        MAPD_col_05 = np.append(MAPD_col_05, MAPD)
        MAE_col_05 = np.append(MAE_col_05, MAE)
        MSE_col_05 = np.append(MSE_col_05, MSE)

    if noise_multiplier == noise_multipliers[4]:
      try:
        epsilons_005 = np.append(epsilons_005,max(math.log((1 - (1/N) - FPR)/FNR), math.log((1 - (1/N) - FNR)/FPR)))
        print("empirical epsilon = " + str(max(math.log((1 - (1/N) - FPR)/FNR), math.log((1 - (1/N) - FNR)/FPR))))
        MAPD_col_005 = np.append(MAPD_col_005, MAPD)
        MAE_col_005 = np.append(MAE_col_005, MAE)
        MSE_col_005 = np.append(MSE_col_005, MSE)
      except:
        print("undefined privacy risk")
        epsilons_005 = np.append(epsilons_005, 0)
        print("empirical epsilon = " + str(0))
        MAPD_col_005 = np.append(MAPD_col_005, MAPD)
        MAE_col_005 = np.append(MAE_col_005, MAE)
        MSE_col_005 = np.append(MSE_col_005, MSE)

end_time = time.time()
elapsed_time = end_time - start_time
print(elapsed_time)

epsilons_13.mean()

epsilons_3.mean()

epsilons_1.mean()

epsilons_05.mean()

epsilons_005.mean()

np.savetxt("epsilons_13_300.csv", epsilons_13, delimiter=",")
np.savetxt("MAPD_13_300.csv", MAPD_col_13, delimiter=",")
np.savetxt("MAE_13_300.csv", MAE_col_13, delimiter=",")
np.savetxt("MSE_13_300.csv", MSE_col_13, delimiter=",")

np.savetxt("epsilons_3_300.csv", epsilons_3, delimiter=",")
np.savetxt("MAPD_3_300.csv", MAPD_col_3, delimiter=",")
np.savetxt("MAE_3_300.csv", MAE_col_3, delimiter=",")
np.savetxt("MSE_3_300.csv", MSE_col_3, delimiter=",")

np.savetxt("epsilons_1_300.csv", epsilons_1, delimiter=",")
np.savetxt("MAPD_1_300.csv", MAPD_col_1, delimiter=",")
np.savetxt("MAE_1_300.csv", MAE_col_1, delimiter=",")
np.savetxt("MSE_1_300.csv", MSE_col_1, delimiter=",")

np.savetxt("epsilons_05_300.csv", epsilons_05, delimiter=",")
np.savetxt("MAPD_05_300.csv", MAPD_col_05, delimiter=",")
np.savetxt("MAE_05_300.csv", MAE_col_05, delimiter=",")
np.savetxt("MSE_05_300.csv", MSE_col_05, delimiter=",")

np.savetxt("epsilons_005_300.csv", epsilons_005, delimiter=",")
np.savetxt("MAPD_005_300.csv", MAPD_col_005, delimiter=",")
np.savetxt("MAE_005_300.csv", MAE_col_005, delimiter=",")
np.savetxt("MSE_005_300.csv", MSE_col_005, delimiter=",")

"""# MLP Schneider et al.

## real
"""

from sklearn.linear_model import LogisticRegression
import statistics

random.seed(1)
np.random.seed(1)
tf.random.set_seed(1)

samples = [100,1000,10000]
#samples = [10000]
churn = pd.read_csv('data.csv', sep = ',', na_values=['(NA)']).fillna(0)
churn = pd.DataFrame.drop_duplicates(churn)
churn['id'] = range(len(churn))

for s in samples:
  print(s)
  random.seed(s)
  np.random.seed(s)
  tf.random.set_seed(s)
  start_time = time.time()
  churn_iter = churn.sample(s)
  id = range(len(churn_iter))

  clf = LogisticRegression(multi_class = 'multinomial', random_state=0, n_jobs = -1).fit(churn_iter.iloc[:,0:9], id)
  print("sample size = " + str(samples))
  print("MLP by Schneider et al = " + str(-1 + math.sqrt(s * sum(np.diagonal(clf.predict_proba(churn_iter.iloc[:,0:9])**2)))))

  end_time = time.time()
  elapsed_time = end_time - start_time
  print("elapsed time = " + str(elapsed_time))

"""## swapping 25%"""

from sklearn.linear_model import LogisticRegression
import statistics

random.seed(1)
np.random.seed(1)
tf.random.set_seed(1)

epsilons = np.array([])
MAPD_col = np.array([])
MAE_col = np.array([])
MSE_col = np.array([])
samples = [100,1000,10000]
churn = pd.read_csv('data.csv', sep = ',', na_values=['(NA)']).fillna(0)
churn = pd.DataFrame.drop_duplicates(churn)

for s in samples:
  print(s)
  random.seed(s)
  np.random.seed(s)
  tf.random.set_seed(s)
  start_time = time.time()
  churn_iter = churn.sample(s)
  id = range(len(churn_iter))

  # protection
  swap_25 = swapping(percent = 0.25, data = churn_iter)

  clf = LogisticRegression(multi_class = 'multinomial', random_state=0, n_jobs = -1).fit(swap_25.iloc[:,0:9], id)
  print("sample size = " + str(samples))
  print("MLP by Schneider et al = " + str(-1 + math.sqrt(s * sum(np.diagonal(clf.predict_proba(churn_iter.iloc[:,0:9])**2)))))

  end_time = time.time()
  elapsed_time = end_time - start_time
  print("elapsed time = " + str(elapsed_time))

"""## swapping 50%"""

from sklearn.linear_model import LogisticRegression
import statistics

random.seed(1)
np.random.seed(1)
tf.random.set_seed(1)

epsilons = np.array([])
MAPD_col = np.array([])
MAE_col = np.array([])
MSE_col = np.array([])
samples = [100,1000,10000]
churn = pd.read_csv('data.csv', sep = ',', na_values=['(NA)']).fillna(0)
churn = pd.DataFrame.drop_duplicates(churn)

for s in samples:
  print(s)
  random.seed(s)
  np.random.seed(s)
  tf.random.set_seed(s)
  start_time = time.time()
  churn_iter = churn.sample(s)
  id = range(len(churn_iter))

  # protection
  swap_50 = swapping(percent = 0.5, data = churn_iter)

  clf = LogisticRegression(multi_class = 'multinomial', random_state=0, n_jobs = -1).fit(swap_50, id)
  print("sample size = " + str(samples))
  print("MLP by Schneider et al = " + str(-1 + math.sqrt(s * sum(np.diagonal(clf.predict_proba(churn_iter)**2)))))

  end_time = time.time()
  elapsed_time = end_time - start_time
  print("elapsed time = " + str(elapsed_time))

"""## copula"""

!pip install copulas --quiet
## might require runtime restart after execution

from sklearn.linear_model import LogisticRegression
from copulas.multivariate import GaussianMultivariate
from copulas.univariate import GaussianUnivariate, GammaUnivariate, BetaUnivariate, GaussianKDE
import statistics

random.seed(1)
np.random.seed(1)
tf.random.set_seed(1)

epsilons = np.array([])
MAPD_col = np.array([])
MAE_col = np.array([])
MSE_col = np.array([])
samples = [10000]
churn = pd.read_csv('data.csv', sep = ',', na_values=['(NA)']).fillna(0)
churn = pd.DataFrame.drop_duplicates(churn)


for s in samples:
  print(s)
  random.seed(s)
  np.random.seed(s)
  tf.random.set_seed(s)
  start_time = time.time()
  churn_iter = churn.sample(s)
  id = range(len(churn_iter))

  # protection
  from sklearn.preprocessing import MinMaxScaler
  scaler0 = MinMaxScaler(feature_range= (-1, 1))
  scaler0 = scaler0.fit(churn_iter)
  churn_scaled = scaler0.transform(churn_iter)
  churn_scaled = pd.DataFrame(churn_scaled)

  # copula train
  dist = GaussianMultivariate(distribution=GaussianUnivariate)
  dist.fit(churn_scaled)
  train_copula = dist.sample(s)
  train_copula = pd.DataFrame(scaler0.inverse_transform(train_copula))
  train_copula.columns = train_copula.columns

  clf = LogisticRegression(multi_class = 'multinomial', random_state=0, n_jobs = -1).fit(train_copula, id)
  print("sample size = " + str(samples))

  print("MLP by Schneider et al = " + str(-1 + math.sqrt(s * sum(np.diagonal(clf.predict_proba(churn_iter)**2)))))

  end_time = time.time()
  elapsed_time = end_time - start_time
  print("elapsed time = " + str(elapsed_time))

"""## anand and lee"""

class GAN():
    def __init__(self, privacy):
      self.img_rows = 1
      self.img_cols = 1
      self.img_shape = (self.img_cols,)
      self.latent_dim = (1)

      optimizer = keras.optimizers.Adam()
      self.discriminator = self.build_discriminator()
      self.discriminator.compile(loss='binary_crossentropy',
                                 optimizer=optimizer,
                                 metrics=['accuracy'])
      if privacy == True:
        print("using differential privacy")
        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(optimizer=DPKerasAdamOptimizer(
            l2_norm_clip=l2_norm_clip,
            noise_multiplier=noise_multiplier,
            num_microbatches=num_microbatches,
            learning_rate=lr),
            loss= tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.losses.Reduction.NONE), metrics=['accuracy'])

      # Build the generator
      self.generator = self.build_generator()

      # The generator takes noise as input and generates imgs
      z = Input(shape=(self.latent_dim,))
      img = self.generator(z)

      # For the combined model we will only train the generator
      self.discriminator.trainable = False

      # The discriminator takes generated images as input and determines validity
      valid = self.discriminator(img)

      # The combined model  (stacked generator and discriminator)
      # Trains the generator to fool the discriminator
      self.combined = Model(z, valid)
      self.combined.compile(loss='binary_crossentropy', optimizer= optimizer)


    def build_generator(self):
      model = Sequential()
      model.add(Dense(self.latent_dim, input_dim=self.latent_dim))
      model.add(LeakyReLU(alpha=0.2))
      #model.add(BatchNormalization())
      model.add(Dense(64, input_shape=self.img_shape))
      model.add(LeakyReLU(alpha=0.2))
      #model.add(BatchNormalization())
      model.add(Dense(self.latent_dim))
      model.add(Activation("tanh"))

      #model.summary()

      noise = Input(shape=(self.latent_dim,))
      img = model(noise)
      return Model(noise, img)

    def build_discriminator(self):

        model = Sequential()

        model.add(Dense(64, input_shape=self.img_shape))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1, activation='sigmoid'))

        #model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)

    def train(self, data, iterations, batch_size, model_name, generator_losses = [], discriminator_acc = [], correlations = [], accuracy = [], MAPD_col = [],MSE_col = [], MAE_col = []):
      # Adversarial ground truths

      valid = np.ones((batch_size, 1))
      fake = np.zeros((batch_size, 1))
      corr = 0
      MAPD = 0
      MSE = 0
      MAE = 0
      #fake += 0.05 * np.random.random(fake.shape)
      #valid += 0.05 * np.random.random(valid.shape)

      for epoch in range(iterations):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random batch of images
            idx = np.random.randint(0, data.shape[0], batch_size)
            imgs = data[idx]

            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

            # Generate a batch of new images
            gen_imgs = self.generator.predict(noise, verbose = False)

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------
            # Train the generator (to have the discriminator label samples as valid)

            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            g_loss = self.combined.train_on_batch(noise, valid)

            # collect losses
            discriminator_acc = np.append(discriminator_acc, 100*d_loss[1])
            generator_losses = np.append(generator_losses, g_loss)
      self.generator.save(model_name)
              #print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f, corr: %f, MAPD: %f, MSE: %f, MAE: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss, corr, MAPD, MSE, MAE))

from sklearn.linear_model import LogisticRegression
import statistics

random.seed(1)
np.random.seed(1)
tf.random.set_seed(1)

samples = [10000]
churn = pd.read_csv('data.csv', sep = ',', na_values=['(NA)']).fillna(0)
churn = pd.DataFrame.drop_duplicates(churn)

for s in samples:
  print(s)
  random.seed(s)
  np.random.seed(s)
  tf.random.set_seed(s)
  start_time = time.time()
  churn_iter = churn.sample(s)
  id =  range(len(churn_iter))

  # protection
  train_outcome = churn_iter[['Tenure']]
  churn_without_tenure = churn_iter.drop('Tenure', axis=1)

  scaler0 = MinMaxScaler(feature_range= (-1, 1))
  scaler0 = scaler0.fit(train_outcome)
  train_outcome1 = scaler0.transform(train_outcome)
  train_outcome1 = pd.DataFrame(train_outcome1)

  print("start train set training")
  gan_train = GAN(privacy = False)
  gan_train.train(data = np.array(train_outcome1), iterations=iterations, batch_size=batch_size, model_name = "train_anand.h5")

  # Generate a batch of new customers
  generator = load_model('train_anand.h5', compile = True)
  noise = np.random.normal(0, 1, (len(train_outcome), 1))
  gen_imgs = generator.predict(noise, verbose = False)
  gen_imgs = scaler0.inverse_transform(gen_imgs)
  gen_imgs = gen_imgs.reshape(len(train_outcome), 1)
  train_GAN = pd.DataFrame(gen_imgs)

  # combine one protected variable with other
  train_GAN.rename(columns = {0:'Tenure'}, inplace = True)
  train = pd.concat([churn_without_tenure.reset_index(drop = True), train_GAN], axis=1)
  train.columns = churn_iter.columns.values

  clf = LogisticRegression(multi_class = 'multinomial', random_state=0, n_jobs = -1).fit(train, id)
  print("sample size = " + str(samples))
  print("MLP by Schneider et al = " + str(-1 + math.sqrt(s * sum(np.diagonal(clf.predict_proba(churn_iter)**2)))))

  end_time = time.time()
  elapsed_time = end_time - start_time
  print("elapsed time = " + str(elapsed_time))

"""## gans with differential privacy"""

!pip install tensorflow_privacy --quiet

class GAN():
    def __init__(self, privacy):
      self.img_rows = 1
      self.img_cols = 9
      self.img_shape = (self.img_cols,)
      self.latent_dim = (9)
      lr = 0.001

      optimizer = keras.optimizers.Adam()
      self.discriminator = self.build_discriminator()
      self.discriminator.compile(loss='binary_crossentropy',
                                 optimizer=optimizer,
                                 metrics=['accuracy'])
      if privacy == True:
        print(noise_multiplier)
        print("using differential privacy")
        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(optimizer=DPKerasAdamOptimizer(
            l2_norm_clip=4,
            noise_multiplier=noise_multiplier,
            num_microbatches=num_microbatches,
            learning_rate=lr),
            loss= tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.losses.Reduction.NONE), metrics=['accuracy'])

      # Build the generator
      self.generator = self.build_generator()

      # The generator takes noise as input and generates imgs
      z = Input(shape=(self.latent_dim,))
      img = self.generator(z)

      # For the combined model we will only train the generator
      self.discriminator.trainable = False

      # The discriminator takes generated images as input and determines validity
      valid = self.discriminator(img)

      # The combined model  (stacked generator and discriminator)
      # Trains the generator to fool the discriminator
      self.combined = Model(z, valid)
      self.combined.compile(loss='binary_crossentropy', optimizer= optimizer)


    def build_generator(self):
      model = Sequential()
      model.add(Dense(self.latent_dim, input_dim=self.latent_dim))
      model.add(LeakyReLU(alpha=0.2))
      #model.add(BatchNormalization())
      model.add(Dense(4, input_shape=self.img_shape))
      model.add(LeakyReLU(alpha=0.2))
      #model.add(BatchNormalization())
      model.add(Dense(self.latent_dim))
      model.add(Activation("tanh"))

      #model.summary()

      noise = Input(shape=(self.latent_dim,))
      img = model(noise)
      return Model(noise, img)

    def build_discriminator(self):

        model = Sequential()

        model.add(Dense(4, input_shape=self.img_shape))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1, activation='sigmoid'))

        #model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)

    def train(self, data, iterations, batch_size, sample_interval, model_name, generator_losses = [], discriminator_acc = [], correlations = [], accuracy = [], MAPD_collect = [],MSE_collect = [], MAE_collect = []):
      # Adversarial ground truths
      valid = np.ones((batch_size, 1))
      fake = np.zeros((batch_size, 1))
      corr = 0
      MAPD = 0
      MSE = 0
      MAE = 0
      #fake += 0.05 * np.random.random(fake.shape)
      #valid += 0.05 * np.random.random(valid.shape)

      for epoch in range(iterations):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random batch of images
            idx = np.random.randint(0, data.shape[0], batch_size)
            imgs = data[idx]

            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

            # Generate a batch of new images
            gen_imgs = self.generator.predict(noise, verbose = False)

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------
            # Train the generator (to have the discriminator label samples as valid)

            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            g_loss = self.combined.train_on_batch(noise, valid)

            if (epoch % 100) == 0:
              print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

      print("save model")
      self.generator.save(model_name)

from keras.models import load_model
from absl import app
from absl import flags
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import logging
from tensorflow_privacy.privacy.analysis import compute_dp_sgd_privacy
from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras import DPKerasSGDOptimizer, DPKerasAdamOptimizer
from tensorflow_privacy.privacy.analysis.compute_dp_sgd_privacy_lib import compute_dp_sgd_privacy
from sklearn.preprocessing import MinMaxScaler

"""### n = 100"""

random.seed(1)
np.random.seed(1)
tf.random.set_seed(1)

start_time = time.time()

churn = pd.read_csv('data.csv', sep = ',', na_values=['(NA)']).fillna(0)
churn = pd.DataFrame.drop_duplicates(churn)

epsilons_inf = np.array([])
MAPD_col_inf = np.array([])
MAE_col_inf = np.array([])
MSE_col_inf = np.array([])

epsilons_13 = np.array([])
MAPD_col_13 = np.array([])
MAE_col_13 = np.array([])
MSE_col_13 = np.array([])

epsilons_3 = np.array([])
MAPD_col_3 = np.array([])
MAE_col_3 = np.array([])
MSE_col_3 = np.array([])

epsilons_1 = np.array([])
MAPD_col_1 = np.array([])
MAE_col_1 = np.array([])
MSE_col_1 = np.array([])

epsilons_05 = np.array([])
MAPD_col_05 = np.array([])
MAE_col_05 = np.array([])
MSE_col_05 = np.array([])

epsilons_005 = np.array([])
MAPD_col_005 = np.array([])
MAE_col_005 = np.array([])
MSE_col_005 = np.array([])

epsilons_001 = np.array([])
MAPD_col_001 = np.array([])
MAE_col_001 = np.array([])
MSE_col_001 = np.array([])

# for 300 obs, 10 iteraties, 100 batch size, epochs = 10
noise_multipliers = [1.011, 2.98, 6.96, 11.85, 60] # 10 iterations

# for 3,000 obs, 100 iteraties, 100 batch size, epochs = 10
#noise_multipliers = [0.6785, 1.43, 3.1, 5.4, 38.5] # 100 iterations, n = 3000
#noise_multipliers = [0.6785]

# for 30,000 obs, 1000 iteraties, 100 batch size, epochs = 10.
#noise_multipliers = [0.502, 0.81, 1.3505, 2.231, 15.3] # 1000 iterations, n = 30000
#noise_multipliers = [0.505]

for iter in range(0,1):
  random.seed(iter)
  np.random.seed(iter)
  tf.random.set_seed(iter)

  for noise in noise_multipliers:
    random.seed(iter)
    np.random.seed(iter)
    tf.random.set_seed(iter)

    # setting epsilon
    N = 100
    batch_size = 100
    iterations = 10
    epochs = iterations/(N/batch_size)
    print("the number of epochs is " + str(epochs))

    print("iteration is " + str(iter))
    churn_iter = churn.sample(100)
    churn_iter = pd.DataFrame(churn_iter)

    scaler0 = MinMaxScaler(feature_range= (-1, 1))
    scaler0 = scaler0.fit(churn_iter)
    train_GAN = scaler0.transform(churn_iter)
    train_GAN = pd.DataFrame(train_GAN)
    id =  range(len(train_GAN))

    noise_multiplier = noise
    ## if l2_norm_clip < ||gradient||2, gradient is preserved. if l2_norm_clip > ||gradient||2, then gradient is divided by l2_norm_clip
    l2_norm_clip = 4 # see abadi et al 2016
    delta= 1/N
    theor_epsilon =compute_dp_sgd_privacy(N, batch_size, noise_multiplier,
                          epochs, delta)
    num_microbatches = batch_size
    print("theoretical epsilon = " + str(round(theor_epsilon[0],2)))

    gan_train = GAN(privacy = True)
    gan_train.train(data = np.array(train_GAN), iterations=iterations, batch_size=batch_size, sample_interval=((iterations-1)/10), model_name = "train_extra.h5")

    generator = load_model('train_extra.h5')

    noise = np.random.normal(0, 1, (N, 9))
    gen_imgs = generator.predict(noise, verbose = False)
    gen_imgs = scaler0.inverse_transform(gen_imgs)
    train_GAN = pd.DataFrame(gen_imgs.reshape(N, 9))
    train_GAN.columns = train.columns.values

    clf = LogisticRegression(multi_class = 'multinomial', random_state=0, n_jobs = -1).fit(train_GAN, id)
    print("MLP by Schneider et al = " + str(-1 + math.sqrt(s * sum(np.diagonal(clf.predict_proba(churn_iter)**2)))))

    if noise_multiplier == noise_multipliers[0]:
      try:
        epsilons_13 = np.append(epsilons_13,max(math.log((1 - FPR)/FNR), math.log((1 - FNR)/FPR)))
        #print("empirical epsilon = " + str(max(math.log((1 - FPR)/FNR), math.log((1 - FNR)/FPR))))
        MAPD_col_13 = np.append(MAPD_col_13, MAPD)
        MAE_col_13 = np.append(MAE_col_13, MAE)
        MSE_col_13 = np.append(MSE_col_13, MSE)
      except:
        epsilons_13 = np.append(epsilons_13, "log error")

    if noise_multiplier == noise_multipliers[1]:
      try:
        epsilons_3 = np.append(epsilons_3,max(math.log((1 - FPR)/FNR), math.log((1 - FNR)/FPR)))
        #print("empirical epsilon = " + str(max(math.log((1 - FPR)/FNR), math.log((1 - FNR)/FPR))))
        MAPD_col_3 = np.append(MAPD_col_3, MAPD)
        MAE_col_3 = np.append(MAE_col_3, MAE)
        MSE_col_3 = np.append(MSE_col_3, MSE)
      except:
        epsilons_3 = np.append(epsilons_3, "log error")

    if noise_multiplier == noise_multipliers[2]:
      try:
        epsilons_1 = np.append(epsilons_1,max(math.log((1 - FPR)/FNR), math.log((1 - FNR)/FPR)))
        #print("empirical epsilon = " + str(max(math.log((1 - FPR)/FNR), math.log((1 - FNR)/FPR))))
        MAPD_col_1 = np.append(MAPD_col_1, MAPD)
        MAE_col_1 = np.append(MAE_col_1, MAE)
        MSE_col_1 = np.append(MSE_col_1, MSE)
      except:
        epsilons_1 = np.append(epsilons_1, "log error")

    if noise_multiplier == noise_multipliers[3]:
      try:
        epsilons_05 = np.append(epsilons_05,max(math.log((1 - FPR)/FNR), math.log((1 - FNR)/FPR)))
        #print("empirical epsilon = " + str(max(math.log((1 - FPR)/FNR), math.log((1 - FNR)/FPR))))
        MAPD_col_05 = np.append(MAPD_col_05, MAPD)
        MAE_col_05 = np.append(MAE_col_05, MAE)
        MSE_col_05 = np.append(MSE_col_05, MSE)
      except:
        epsilons_05 = np.append(epsilons_05, "log error")

    if noise_multiplier == noise_multipliers[4]:
      try:
        epsilons_005 = np.append(epsilons_005,max(math.log((1 - FPR)/FNR), math.log((1 - FNR)/FPR)))
        #print("empirical epsilon = " + str(max(math.log((1 - FPR)/FNR), math.log((1 - FNR)/FPR))))
        MAPD_col_005 = np.append(MAPD_col_005, MAPD)
        MAE_col_005 = np.append(MAE_col_005, MAE)
        MSE_col_005 = np.append(MSE_col_005, MSE)
      except:
        epsilons_005 = np.append(epsilons_005, "log error")

end_time = time.time()
elapsed_time = end_time - start_time
print(elapsed_time)

"""### n = 1000"""

random.seed(1)
np.random.seed(1)
tf.random.set_seed(1)

start_time = time.time()

churn = pd.read_csv('data.csv', sep = ',', na_values=['(NA)']).fillna(0)
churn = pd.DataFrame.drop_duplicates(churn)

epsilons_inf = np.array([])
MAPD_col_inf = np.array([])
MAE_col_inf = np.array([])
MSE_col_inf = np.array([])

epsilons_13 = np.array([])
MAPD_col_13 = np.array([])
MAE_col_13 = np.array([])
MSE_col_13 = np.array([])

epsilons_3 = np.array([])
MAPD_col_3 = np.array([])
MAE_col_3 = np.array([])
MSE_col_3 = np.array([])

epsilons_1 = np.array([])
MAPD_col_1 = np.array([])
MAE_col_1 = np.array([])
MSE_col_1 = np.array([])

epsilons_05 = np.array([])
MAPD_col_05 = np.array([])
MAE_col_05 = np.array([])
MSE_col_05 = np.array([])

epsilons_005 = np.array([])
MAPD_col_005 = np.array([])
MAE_col_005 = np.array([])
MSE_col_005 = np.array([])

epsilons_001 = np.array([])
MAPD_col_001 = np.array([])
MAE_col_001 = np.array([])
MSE_col_001 = np.array([])

# for 300 obs, 10 iteraties, 100 batch size, epochs = 10
#noise_multipliers = [1.011, 2.98, 6.96, 11.85, 60] # 10 iterations

# for 3,000 obs, 100 iteraties, 100 batch size, epochs = 10
noise_multipliers = [0.6785, 1.43, 3.1, 5.4, 38.5] # 100 iterations, n = 3000

# for 30,000 obs, 1000 iteraties, 100 batch size, epochs = 10.
#noise_multipliers = [0.502, 0.81, 1.3505, 2.231, 15.3] # 1000 iterations, n = 30000
#noise_multipliers = [0.505]

for noise in noise_multipliers:
  random.seed(1)
  np.random.seed(1)
  tf.random.set_seed(1)

  # setting epsilon
  N = 1000
  batch_size = 100
  iterations = 100
  epochs = iterations/(N/batch_size)
  print("the number of epochs is " + str(epochs))

  print("iteration is " + str(iter))
  churn_iter = churn.sample(N)
  churn_iter = pd.DataFrame(churn_iter)

  scaler0 = MinMaxScaler(feature_range= (-1, 1))
  scaler0 = scaler0.fit(churn_iter)
  train_GAN = scaler0.transform(churn_iter)
  train_GAN = pd.DataFrame(train_GAN)

  id =  range(len(train_GAN))

  noise_multiplier = noise

  ## if l2_norm_clip < ||gradient||2, gradient is preserved. if l2_norm_clip > ||gradient||2, then gradient is divided by l2_norm_clip
  l2_norm_clip = 4 # see abadi et al 2016
  delta= 1/N
  theor_epsilon =compute_dp_sgd_privacy(N, batch_size, noise_multiplier,
                          epochs, delta)
  num_microbatches = batch_size
  print("theoretical epsilon = " + str(round(theor_epsilon[0],2)))

  gan_train = GAN(privacy = True)
  gan_train.train(data = np.array(train_GAN), iterations=iterations, batch_size=batch_size, sample_interval=((iterations-1)/10), model_name = "train_extra.h5")

  generator = load_model('train_extra.h5')

  noise = np.random.normal(0, 1, (N, 9))
  gen_imgs = generator.predict(noise, verbose = False)
  gen_imgs = scaler0.inverse_transform(gen_imgs)
  train_GAN = pd.DataFrame(gen_imgs.reshape(N, 9))
  train_GAN.columns = train.columns.values

  clf = LogisticRegression(multi_class = 'multinomial', random_state=0, n_jobs = -1).fit(train_GAN, id)
  print("MLP by Schneider et al = " + str(-1 + math.sqrt(s * sum(np.diagonal(clf.predict_proba(churn_iter)**2)))))

end_time = time.time()
elapsed_time = end_time - start_time
print(elapsed_time)

"""### n = 10,000"""

random.seed(1)
np.random.seed(1)
tf.random.set_seed(1)

start_time = time.time()

churn = pd.read_csv('data.csv', sep = ',', na_values=['(NA)']).fillna(0)
churn = pd.DataFrame.drop_duplicates(churn)

epsilons_inf = np.array([])
MAPD_col_inf = np.array([])
MAE_col_inf = np.array([])
MSE_col_inf = np.array([])

epsilons_13 = np.array([])
MAPD_col_13 = np.array([])
MAE_col_13 = np.array([])
MSE_col_13 = np.array([])

epsilons_3 = np.array([])
MAPD_col_3 = np.array([])
MAE_col_3 = np.array([])
MSE_col_3 = np.array([])

epsilons_1 = np.array([])
MAPD_col_1 = np.array([])
MAE_col_1 = np.array([])
MSE_col_1 = np.array([])

epsilons_05 = np.array([])
MAPD_col_05 = np.array([])
MAE_col_05 = np.array([])
MSE_col_05 = np.array([])

epsilons_005 = np.array([])
MAPD_col_005 = np.array([])
MAE_col_005 = np.array([])
MSE_col_005 = np.array([])

epsilons_001 = np.array([])
MAPD_col_001 = np.array([])
MAE_col_001 = np.array([])
MSE_col_001 = np.array([])

# for 300 obs, 10 iteraties, 100 batch size, epochs = 10
#noise_multipliers = [1.011, 2.98, 6.96, 11.85, 60] # 10 iterations

# for 3,000 obs, 100 iteraties, 100 batch size, epochs = 10
#noise_multipliers = [0.6785, 1.43, 3.1, 5.4, 38.5] # 100 iterations, n = 3000

# for 30,000 obs, 1000 iteraties, 100 batch size, epochs = 10.
noise_multipliers = [0.502, 0.81, 1.3505, 2.231, 15.3] # 1000 iterations, n = 30000

for noise in noise_multipliers:
  random.seed(1)
  np.random.seed(1)
  tf.random.set_seed(1)

  # setting epsilon
  N = 10000
  batch_size = 100
  iterations = 1000
  epochs = iterations/(N/batch_size)
  print("the number of epochs is " + str(epochs))

  print("iteration is " + str(iter))
  churn_iter = churn.sample(N)
  churn_iter = pd.DataFrame(churn_iter)

  scaler0 = MinMaxScaler(feature_range= (-1, 1))
  scaler0 = scaler0.fit(churn_iter)
  train_GAN = scaler0.transform(churn_iter)
  train_GAN = pd.DataFrame(train_GAN)

  id =  range(len(train_GAN))

  noise_multiplier = noise

  ## if l2_norm_clip < ||gradient||2, gradient is preserved. if l2_norm_clip > ||gradient||2, then gradient is divided by l2_norm_clip
  l2_norm_clip = 4 # see abadi et al 2016
  delta= 1/N
  theor_epsilon =compute_dp_sgd_privacy(N, batch_size, noise_multiplier,
                          epochs, delta)
  num_microbatches = batch_size
  print("theoretical epsilon = " + str(round(theor_epsilon[0],2)))

  gan_train = GAN(privacy = True)
  gan_train.train(data = np.array(train_GAN), iterations=iterations, batch_size=batch_size, sample_interval=((iterations-1)/10), model_name = "train_extra.h5")

  generator = load_model('train_extra.h5')

  noise = np.random.normal(0, 1, (N, 9))
  gen_imgs = generator.predict(noise, verbose = False)
  gen_imgs = scaler0.inverse_transform(gen_imgs)
  train_GAN = pd.DataFrame(gen_imgs.reshape(N, 9))
  train_GAN.columns = train.columns.values

  clf = LogisticRegression(multi_class = 'multinomial', random_state=0, n_jobs = -1).fit(train_GAN, id)
  print("MLP by Schneider et al = " + str(-1 + math.sqrt(s * sum(np.diagonal(clf.predict_proba(churn_iter)**2)))))

end_time = time.time()
elapsed_time = end_time - start_time
print(elapsed_time)
