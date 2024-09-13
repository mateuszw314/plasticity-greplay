#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import tensorflow as tf
import tensorflow.keras.layers as tfl
import datagen
from cbp_linear import CBPLinear

# In[2]:


@tf.keras.saving.register_keras_serializable()
class Sampling(tfl.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


# In[3]:


@tf.keras.saving.register_keras_serializable()
class CIFARVAE(tf.keras.Model):
    def __init__(self, alpha=10, conditional=False, **kwargs):
        super().__init__(**kwargs)

        self.encoder = self.get_encoder()
        self.decoder = self.get_decoder()
        self.alpha = alpha
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.conditional = conditional
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")
        if self.conditional:
            self.prior_means = tf.Variable([tf.random.normal((100,)) for ii in range(100)],
                                           name='prior_means')
            self.prior_logvars = tf.Variable([tf.random.normal((100,)) for ii in range(100)],
                                             name='prior_logvars')

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def get_encoder(self, latent_dim=100):
        picture_inputs = tfl.Input(shape=(32, 32, 3))
        feature_extractor = tf.keras.models.load_model(
           '/home/users/mateuwa/PycharmProjects/cil-rejection-sampling/models/feature_extractor_conv.keras')
        feature_extractor.trainable = False
        #x = tfl.RandomRotation(0.2)(x)
        #x = tfl.RandomFlip()(x)
        x = feature_extractor(picture_inputs)
        x = tfl.Flatten()(x)
        x = tfl.Dense(2000, activation=tfl.ReLU())(x)
        x = tfl.Dense(2000, activation=tfl.ReLU())(x)

        z_mean = tfl.Dense(latent_dim, name="z_mean")(x)
        z_log_var = tfl.Dense(latent_dim, name="z_log_var")(x)
        z = Sampling()([z_mean, z_log_var])
        return tf.keras.Model(picture_inputs, [z_mean, z_log_var, z], name="encoder")

    def get_decoder(self, latent_dim=100):
        return tf.keras.Sequential(
            [
                tfl.Dense(units=2000, activation=None),
                tfl.BatchNormalization(),
                tfl.ReLU(),

                tfl.Dense(units=4 * 4 * 256, activation=None),
                tfl.BatchNormalization(),
                tfl.ReLU(),

                tfl.Reshape((4, 4, 256)),

                tfl.Conv2DTranspose(128, 4, strides=2, padding='same', activation=None),
                tfl.BatchNormalization(),
                tfl.ReLU(),

                tfl.Conv2DTranspose(64, 4, strides=2, padding='same', activation=None),
                tfl.BatchNormalization(),
                tfl.ReLU(),

                tfl.Conv2DTranspose(32, 4, strides=2, padding='same', activation=None),
                tfl.BatchNormalization(),
                tfl.ReLU(),

                tfl.Conv2DTranspose(16, 4, strides=1, padding='same', activation=None),
                tfl.BatchNormalization(),
                tfl.ReLU(),

                tfl.Conv2DTranspose(3, 3, strides=1, padding='same', activation=None),
            ]
        )

    def train_step(self, data):
        with tf.GradientTape() as tape:
            images, labels = data
            z_mean, z_log_var, z = self.encoder(images)
            reconstruction = self.decoder(z)
            labels = tf.argmax(labels, axis=1)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    tf.keras.losses.mse(images, reconstruction), axis=(1, 2)
                )
            )

            if self.conditional:
                tmp_prior_means = tf.gather(self.prior_means, labels)
                tmp_prior_logvars = tf.gather(self.prior_logvars, labels)
                kl_loss = 1 + z_log_var - tmp_prior_logvars - (1 / tf.exp(tmp_prior_logvars)) * \
                          (tf.square(z_mean - tmp_prior_means) + tf.exp(z_log_var))
                kl_loss *= -0.5
                # kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
                kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            else:
                kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
                kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss * self.alpha + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }


# In[4]:


mirrored_strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())


# In[5]:


with mirrored_strategy.scope():
    generator = CIFARVAE(conditional=True, alpha=1.)
    generator.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=1e-4))


# In[10]:


for task in range(10):
    (X_train, Y_train), (X_test, Y_test) = tf.keras.datasets.cifar100.load_data()
    X_train = X_train[tf.squeeze(Y_train) < (task+1)*10]
    Y_train = Y_train[tf.squeeze(Y_train) < (task+1)*10]
    X_train = (X_train -127.5)/127.5
    X_test = (X_test -127.5)/127.5
    print(X_train.shape)
    generator.fit(X_train, Y_train,  epochs=100, batch_size=256, verbose=1)
    




# In[11]:


generator.save_weights('generator_class_incremental_with_return_alpha_1')






