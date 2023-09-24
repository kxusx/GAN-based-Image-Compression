# generate a gan model

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers
from tensorflow.keras import losses
from tensorflow.keras import metrics
from tensorflow.keras import backend as K

class GAN():
    # GAN model
    def __init__(self, input_shape, latent_dim, generator, discriminator):
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.generator = generator
        self.discriminator = discriminator
        self.discriminator.trainable = False
        self.gan = models.Sequential()
        self.gan.add(generator)
        self.gan.add(discriminator)
        self.gan.compile(optimizer=optimizers.RMSprop(lr=0.0008, clipvalue=1.0, decay=1e-8), loss='binary_crossentropy')
        self.gan.summary()

    def train(self, x_train, batch_size, epochs, save_dir):
        # train gan model
        x_train: training_data
        batch_size: batch_size
        epochs: epochs
        save_dir: save_directory
        # train discriminator
        for epoch in range(epochs):
            print('Epoch: ', epoch)
            print('Discriminator training...')
            # get random real images
            random_index = np.random.randint(0, x_train.shape[0], size=batch_size)
            real_images = x_train[random_index]
            # generate fake images
            random_latent_vectors = np.random.normal(size=(batch_size, self.latent_dim))
            generated_images = self.generator.predict(random_latent_vectors)
            # combine real and fake images
            combined_images = np.concatenate([generated_images, real_images])
            # combine labels
            labels = np.concatenate([np.ones((batch_size, 1)), np.zeros((batch_size, 1))])
            # add random noise to labels
            labels += 0.05 * np.random.random(labels.shape)
            # train discriminator
            d_loss = self.discriminator.train_on_batch(combined_images, labels)
            print('d_loss: ', d_loss)
            # train generator
            print('Generator training...')
            random_latent_vectors = np.random.normal(size=(batch_size, self.latent_dim))
            misleading_targets = np.zeros((batch_size, 1))
            a_loss = self.gan.train_on_batch(random_latent_vectors, misleading_targets)
            print('a_loss: ', a_loss)
            # save images
            if epoch % 100 == 0:
                self.save_images(epoch, save_dir)
            
    def save_images(self, epoch, save_dir):
        # save images
        image_array = np.full(( 
            10 * self.input_shape[0], 
            10 * self.input_shape[1], 
            self.input_shape[2]), 
            255, 
            dtype=np.uint8)
        random_latent_vectors = np.random.normal(size=(100, self.latent_dim))
        generated_images = self.generator.predict(random_latent_vectors)
        generated_images = generated_images * 127.5 + 127.5
        generated_images = generated_images.astype(np.uint8)
        for i in range(10):
            for j in range(10):
                image_array[i * self.input_shape[0] : (i + 1) * self.input_shape[0], j * self.input_shape[1] : (j + 1) * self.input_shape[1]] = generated_images[i * 10 + j]
        image = Image.fromarray(image_array)
        image.save(os.path.join(save_dir, 'generated_img_' + str(epoch) + '.png'))
    
    def generate_images(self, save_dir):
        # generate images
        image_array = np.full(( 
            10 * self.input_shape[0], 
            10 * self.input_shape[1], 
            self.input_shape[2]), 
            255, 
            dtype=np.uint8)
        random_latent_vectors = np.random.normal(size=(100, self.latent_dim))
        generated_images = self.generator.predict(random_latent_vectors)
        generated_images = generated_images * 127.5 + 127.5
        generated_images = generated_images.astype(np.uint8)
        for i in range(10):
            for j in range(10):
                image_array[i * self.input_shape[0] : (i + 1) * self.input_shape[0], j * self.input_shape[1] : (j + 1) * self.input_shape[1]] = generated_images[i * 10 + j]
        image = Image.fromarray(image_array)
        image.save(os.path.join(save_dir, 'generated_img.png'))
    
    def save_model(self, save_dir):
        # save model
        self.generator.save(os.path.join(save_dir, 'generator.h5'))
        self.discriminator.save(os.path.join(save_dir, 'discriminator.h5'))
        self.gan.save(os.path.join(save_dir, 'gan.h5'))

# Path: generator.py
# generate a generator model

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers
from tensorflow.keras import losses
from tensorflow.keras import metrics
from tensorflow.keras import backend as K

class Generator():
    # generator model
    def __init__(self, input_shape, latent_dim):
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.generator = models.Sequential()
        self.generator.add(layers.Dense(128 * 7 * 7, use_bias=False, input_shape=(self.latent_dim,)))
        self.generator.add(layers.BatchNormalization())
        self.generator.add(layers.LeakyReLU())
        self.generator.add(layers.Reshape((7, 7, 128)))
        self.generator.add(layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same', use_bias=False))
        self.generator.add(layers.BatchNormalization())
        self.generator.add(layers.LeakyReLU())
        self.generator.add(layers.Conv2DTranspose(64, (4, 4), strides=(2, 2), padding='same', use_bias=False))
        self.generator.add(layers.BatchNormalization())
        self.generator.add(layers.LeakyReLU())
        self.generator.add(layers.Conv2DTranspose(1, (4, 4), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
        self.generator.summary()

    def generate(self, random_latent_vectors):
        # generate images
        generated_images = self.generator.predict(random_latent_vectors)
        return generated_images
    
    def save(self, save_dir):
        # save model
        self.generator.save(os.path.join(save_dir, 'generator.h5'))
    
    def load(self, load_dir):
        # load model
        self.generator = models.load_model(os.path.join(load_dir, 'generator.h5'))
    
    def get_model(self):
        # get model
        return self.generator
    
    def get_summary(self):
        # get summary
        return self.generator.summary()
    
    def get_config(self):
        # get config
        return self.generator.get_config()
    
    def get_weights(self):
        # get weights
        return self.generator.get_weights()
    
    def set_weights(self, weights):
        # set weights
        self.generator.set_weights(weights)
    
    def compile(self, optimizer, loss, metrics):
        # compile model
        self.generator.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    
    def train_on_batch(self, x, y):
        # train model on batch
        return self.generator.train_on_batch(x, y)
    
    def evaluate(self, x, y, batch_size):
        # evaluate model
        return self.generator.evaluate(x, y, batch_size=batch_size)

# Path: discriminator.py
# generate a discriminator model

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers
from tensorflow.keras import losses
from tensorflow.keras import metrics
from tensorflow.keras import backend as K

class Discriminator():
    # discriminator model
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.discriminator = models.Sequential()
        self.discriminator.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=self.input_shape))
        self.discriminator.add(layers.LeakyReLU())
        self.discriminator.add(layers.Dropout(0.3))
        self.discriminator.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
        self.discriminator.add(layers.LeakyReLU())
        self.discriminator.add(layers.Dropout(0.3))
        self.discriminator.add(layers.Flatten())
        self.discriminator.add(layers.Dense(1))
        self.discriminator.summary()
    
    def discriminate(self, images):
        # discriminate images
        predictions = self.discriminator.predict(images)
        return predictions
    
    def save(self, save_dir):
        # save model
        self.discriminator.save(os.path.join(save_dir, 'discriminator.h5'))
    
    def load(self, load_dir):
        # load model
        self.discriminator = models.load_model(os.path.join(load_dir, 'discriminator.h5'))
    
    def get_model(self):
        # get model
        return self.discriminator
    
    def get_summary(self):
        # get summary
        return self.discriminator.summary()
    
    def get_config(self):
        # get config
        return self.discriminator.get_config()
    
    def get_weights(self):
        # get weights
        return self.discriminator.get_weights()
    
    def set_weights(self, weights):
        # set weights
        self.discriminator.set_weights(weights)
    
    def compile(self, optimizer, loss, metrics):
        # compile model
        self.discriminator.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    
    def train_on_batch(self, x, y):
        # train model on batch
        return self.discriminator.train_on_batch(x, y)
    
    def evaluate(self, x, y, batch_size):
        # evaluate model
        return self.discriminator.evaluate(x, y, batch_size=batch_size)
        
