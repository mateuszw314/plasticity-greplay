import numpy as np
from copy import deepcopy
from keras.datasets import mnist, fashion_mnist, cifar100
from tensorflow import squeeze as tf_squeeze
#import tensorflow_datasets as tfds
from pathlib import Path
import tensorflow as tf
import os
### from Nguyen

class PermutedMnistGenerator():
    def __init__(self, max_iter=10):
        (self.X_train, self.Y_train), (self.X_test, self.Y_test) = mnist.load_data()
        self.X_train = self.X_train.reshape((60000, 784))
        self.X_test = self.X_test.reshape((10000, 784))

        self.max_iter = max_iter
        self.cur_iter = 0


    def get_dims(self):
        # Get data input and output dimensions
        return self.X_train.shape[1], 10

    def next_task(self):
        if self.cur_iter >= self.max_iter:
            raise Exception('Number of tasks exceeded!')
        else:
            np.random.seed(self.cur_iter)
            perm_inds = list(range(self.X_train.shape[1]))
            np.random.shuffle(perm_inds)

            # Retrieve train data
            next_x_train = deepcopy(self.X_train)
            next_x_train = next_x_train[:,perm_inds].astype(np.float32)
            next_y_train = np.eye(10)[self.Y_train].astype(np.float32)

            # Retrieve test data
            next_x_test = deepcopy(self.X_test)
            next_x_test = next_x_test[:,perm_inds].astype(np.float32)
            next_y_test = np.eye(10)[self.Y_test].astype(np.float32)

            self.cur_iter += 1

            return next_x_train/255., next_y_train, next_x_test/255., next_y_test


class SplitMnistGenerator():
    def __init__(self, append_fashion=False):
        (self.X_train, self.Y_train), (self.X_test, self.Y_test) = mnist.load_data()
        self.max_iter = 5
        if append_fashion:
            self.max_iter = 10
            (X_train_f, Y_train_f), (X_test_f, Y_test_f) = fashion_mnist.load_data()
            Y_test_f = Y_test_f + 10
            Y_train_f = Y_train_f + 10
            self.X_train = np.concatenate((self.X_train, X_train_f))
            self.Y_train = np.concatenate((self.Y_train, Y_train_f))
            self.X_test = np.concatenate((self.X_test, X_test_f))
            self.Y_test = np.concatenate((self.Y_test, Y_test_f))
        self.X_train = self.X_train.reshape((self.X_train.shape[0], 784))
        self.X_test = self.X_test.reshape((self.X_test.shape[0], 784))

        self.cur_iter = 0


    def get_dims(self):
        # Get data input and output dimensions
        return self.X_train.shape[1], 10

    def next_task(self):
        if self.cur_iter >= self.max_iter:
            raise Exception('Number of tasks exceeded!')
        else:

            # Retrieve train data
            next_y_train = deepcopy(self.Y_train[np.isin(self.Y_train, [2*self.cur_iter, 2*self.cur_iter+1])])
            next_x_train = deepcopy(self.X_train[np.isin(self.Y_train, [2*self.cur_iter, 2*self.cur_iter+1])]).astype(np.float32)
            next_y_train = np.eye(2*self.max_iter)[next_y_train].astype(np.float32)

            # Retrieve test data
            next_y_test = deepcopy(self.Y_test[np.isin(self.Y_test, [2 * self.cur_iter, 2 * self.cur_iter + 1])])
            next_x_test = deepcopy(self.X_test[np.isin(self.Y_test, [2*self.cur_iter, 2*self.cur_iter+1])]).astype(np.float32)
            next_y_test = np.eye(2*self.max_iter)[next_y_test].astype(np.float32)

            self.cur_iter += 1

            return next_x_train/256., next_y_train, next_x_test/256., next_y_test


class SplitEMnistGenerator():
    def __init__(self, num_tasks=23):

        #(self.X_train, self.Y_train) = tfds.as_numpy(tfds.load('emnist/balanced', as_supervised=True, split=['train'], batch_size=-1)[0])
        #(self.X_test, self.Y_test) = tfds.as_numpy(tfds.load('emnist/balanced', as_supervised=True, split=['test'], batch_size=-1)[0])
        self.max_iter = num_tasks
        #self.X_train = self.X_train.reshape((self.X_train.shape[0], 784))
        #self.X_test = self.X_test.reshape((self.X_test.shape[0], 784))

        self.cur_iter = 0

    #def get_dims(self):
    #    # Get data input and output dimensions
    #    return self.X_train.shape[1], 10

    def next_task(self):
        if self.cur_iter >= self.max_iter:
            raise Exception('Number of tasks exceeded!')
        else:
            (X_train, Y_train) = tfds.as_numpy(
                tfds.load('emnist/balanced', as_supervised=True, split=['train'], batch_size=-1)[0])
            (X_test, Y_test) = tfds.as_numpy(
                tfds.load('emnist/balanced', as_supervised=True, split=['test'], batch_size=-1)[0])
            X_train = X_train.reshape((X_train.shape[0], 784))
            X_test = X_test.reshape((X_test.shape[0], 784))

            # Retrieve train data
            next_y_train = deepcopy(Y_train[np.isin(Y_train, [2 * self.cur_iter, 2 * self.cur_iter + 1])])
            next_x_train = deepcopy(
                X_train[np.isin(Y_train, [2 * self.cur_iter, 2 * self.cur_iter + 1])]).astype(np.float32)
            next_y_train = np.eye(2*self.max_iter)[next_y_train].astype(np.float32)

            # Retrieve test data
            next_y_test = deepcopy(Y_test[np.isin(Y_test, [2 * self.cur_iter, 2 * self.cur_iter + 1])])
            next_x_test = deepcopy(
                X_test[np.isin(Y_test, [2 * self.cur_iter, 2 * self.cur_iter + 1])]).astype(np.float32)
            next_y_test = np.eye(2*self.max_iter)[next_y_test].astype(np.float32)

            self.cur_iter += 1

            return next_x_train / 256., next_y_train, next_x_test / 256., next_y_test


class SplitCIFAR100Generator():
    def __init__(self, num_tasks=20):

        self.max_iter = num_tasks
        self.classes_per_task = 100 // num_tasks
        self.cur_iter = 0

    def next_task(self):
        if self.cur_iter >= self.max_iter:
            raise Exception('Number of tasks exceeded!')
        else:
            (X_train, Y_train), (X_test, Y_test) = cifar100.load_data()
            classes_per_task = self.classes_per_task
            Y_train = tf_squeeze(Y_train)
            Y_test = tf_squeeze(Y_test)
            # Retrieve train data
            classrange = np.arange(self.cur_iter * classes_per_task,
                                   self.cur_iter * classes_per_task + classes_per_task)
            next_y_train = Y_train[np.isin(Y_train, classrange)]
            next_x_train = X_train[np.isin(Y_train, classrange)].astype(np.float32)
            next_y_train = np.eye(self.classes_per_task * self.max_iter)[next_y_train].astype(np.float32)

            # Retrieve test data
            next_y_test = Y_test[np.isin(Y_test, classrange)]
            next_x_test = X_test[np.isin(Y_test, classrange)].astype(np.float32)
            next_y_test = np.eye(self.classes_per_task * self.max_iter)[next_y_test].astype(np.float32)

            self.cur_iter += 1
            next_x_train = (next_x_train - 127.5) / 127.5
            next_x_test = (next_x_test - 127.5) / 127.5
            return next_x_train, next_y_train, next_x_test, next_y_test


class SplitCore50Generator():
    def __init__(self, data_dir, num_tasks=10, batch_size=256, shuffle_classes=False, validation_ratio=0.2):

        self.max_iter = num_tasks
        self.classes_per_task = 50 // num_tasks
        self.cur_iter = 0
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.class_order = np.arange(1, 51)
        if shuffle_classes:
            np.random.shuffle(self.class_order)
        self.validation_ratio = validation_ratio
        #naxt_task zrwaca train_ds i val_ds, cala klasa odpowiada za wybor i zaladowanie odpowiednich klas z datasetu w stosunku do numeru zadania

    def get_label(self, file_path):
        # Convert the path to a list of path components
        parts = tf.strings.split(file_path, os.path.sep)
        # The second to last is the class-directory
        label = tf.strings.substr(parts[-2], 1, 2)
        # Integer encode the label
        return tf.one_hot(int(label), 50)
        
    def decode_img(self, img):
        # Convert the compressed string to a 3D uint8 tensor
        img = tf.io.decode_png(img, channels=3)
        # Resize the image to the desired size
        return img

    def process_path(self, file_path):
        label = self.get_label(file_path)
        # Load the raw data from the file as a string
        img = tf.io.read_file(file_path)
        img = self.decode_img(img)
        return img, label

    def configure_for_performance(self, ds):
        ds = ds.cache()
        ds = ds.shuffle(buffer_size=1000)
        ds = ds.batch(self.batch_size)
        ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
        return ds

    def get_list_dataset(self, classes):
        for ii, data_class in enumerate(classes):
            path = str(Path(f"{self.data_dir}/o{data_class}/*.png"))
            if ii == 0:
                list_ds = tf.data.Dataset.list_files(path, shuffle=False)
            else:
                try:
                    list_ds = list_ds.concatenate(tf.data.Dataset.list_files(path, shuffle=False))
                except:
                    print('Could not read files from', path)# TODO: check if path exists instead of trycatch
        return list_ds


    def next_task(self):
        if self.cur_iter >= self.max_iter:
            raise Exception('Number of tasks exceeded!')
        else:
            start = self.cur_iter*self.classes_per_task
            end = self.cur_iter*self.classes_per_task+self.classes_per_task
            classes = self.class_order[start:end]
            list_ds = self.get_list_dataset(classes)
            image_count = list_ds.cardinality().numpy()
            list_ds = list_ds.shuffle(image_count, reshuffle_each_iteration=False)
            val_size = int(image_count * self.validation_ratio)
            train_ds = list_ds.skip(val_size)
            val_ds = list_ds.take(val_size)
            
            train_ds = train_ds.map(self.process_path, num_parallel_calls=tf.data.AUTOTUNE)
            val_ds = val_ds.map(self.process_path, num_parallel_calls=tf.data.AUTOTUNE)

            
            normalization_layer = tf.keras.layers.Rescaling(1./127.5, offset=-1)
            train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y), num_parallel_calls=tf.data.AUTOTUNE)
            val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y), num_parallel_calls=tf.data.AUTOTUNE)

            train_ds = self.configure_for_performance(train_ds)
            val_ds = self.configure_for_performance(val_ds)
            self.cur_iter += 1
            
            return train_ds, val_ds


