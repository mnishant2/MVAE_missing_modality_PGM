import os
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedShuffleSplit
from data.metadata import metadata
from config import *
from paths import *


class person():
    """
    An entity to define linking connection of entities in each dataset.
    Each entity person has their own language of MNIST scribble and their 
    own unique speaker id defined in metadata.py.

    TODO:
      * Change self._get_mnist() -> self._get_speech().
      * In paths.py correct speech_data_dir. 
        
    """
    def __init__(self, mnist_language_1='Arabic', mnist_language_2='Arabic', dtype=np.float32):
        
        self.mnist_language_1 = mnist_language_1
        self.mnist_language_2 = mnist_language_2
        self.dtype = dtype
        
        # MNIST modality 1
        (self.mnist_1_X_train, self.mnist_1_X_valid, self.mnist_1_X_test, 
         self.mnist_1_y_train, self.mnist_1_y_valid, self.mnist_1_y_test) = self._get_mnist(self.mnist_language_1)
        
        # MNIST modality 2
        (self.mnist_2_X_train, self.mnist_2_X_valid, self.mnist_2_X_test, 
         self.mnist_2_y_train, self.mnist_2_y_valid, self.mnist_2_y_test) = self._get_mnist(self.mnist_language_2)
        
        # Speech modality
        (self.speech_X_train, self.speech_X_valid, self.speech_X_test, 
         self.speech_y_train, self.speech_y_valid, self.speech_y_test) = self._get_speech()

    def _get_train_test_from_npz(self, npz):
        train_test_dict = dict(np.load(npz))
        (X_train, X_test, y_train, y_test) = [train_test_dict[k] for k in ["X_train", "X_test", "y_train", "y_test"]]
        return (X_train, X_test, y_train, y_test)
      
    def _get_stratified_split(self, data_x, data_y, test_ratio=0.2):
        train_index, val_index = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio).split(data_x, data_y).__next__()
        return data_x[train_index], data_x[val_index], data_y[train_index], data_y[val_index]

    def _get_mnist(self, mnist_language):
        # Load train-test from npz
        npz = os.path.join(mnist_data_dir, mnist_language + '_train_test.npz')
        (X_train, X_test, y_train, y_test) = self._get_train_test_from_npz(npz)
        # Convert X to channel-last convention 4D tensor
        (X_train, X_test) = (X_train.reshape(-1, 28, 28, 1), X_test.reshape(-1, 28, 28, 1))
        # Get Train-Valid split
        (X_train, X_valid, y_train, y_valid) = self._get_stratified_split(X_train, y_train)
        # Change dtype
        (X_train, X_valid, X_test, y_train, y_valid, y_test) = (X_train.astype(self.dtype), 
                                                                X_valid.astype(self.dtype), 
                                                                X_test.astype(self.dtype), 
                                                                y_train.astype(self.dtype), 
                                                                y_valid.astype(self.dtype), 
                                                                y_test.astype(self.dtype))

        def normalize(x):
          x = x / 255.0
          return x

        X_train, X_valid, X_test = normalize(X_train), normalize(X_valid), normalize(X_test)

        return (X_train, X_valid, X_test, y_train, y_valid, y_test)

    def _get_speech(self):
        # Initialize list to contain all speaker's datasets
        X_train, X_test = [np.random.rand(0, 1, 13)] * 2
        y_train, y_test = [np.random.rand(0,)] * 2
        for name in ["jackson", "nicolas", "theo", "yweweler", "george", "lucas"]:
            npz = os.path.join(speech_data_dir, name + '_train_test.npz')
            (x_tr, x_ts, y_tr, y_ts) = self._get_train_test_from_npz(npz)
            X_train = np.append(X_train, x_tr, axis=0)
            X_test = np.append(X_test, x_ts, axis=0)
            y_train = np.append(y_train, y_tr, axis=0)
            y_test = np.append(y_test, y_ts, axis=0)
        # Get Train-Valid split
        (X_train, X_valid, y_train, y_valid) = self._get_stratified_split(X_train, y_train)

        X_train, X_valid, X_test = X_train.squeeze(), X_valid.squeeze(), X_test.squeeze()

        def normalize(x):
          X_min = X_train.min(0)
          X_max = X_train.max(0)
          x = (x - X_min) / (X_max - X_min)
          return x

        X_train, X_valid, X_test = normalize(X_train.squeeze()), normalize(X_valid.squeeze()), normalize(X_test.squeeze())

        # Change dtype
        (X_train, X_valid, X_test, y_train, y_valid, y_test) = (X_train.astype(self.dtype), 
                                                                X_valid.astype(self.dtype), 
                                                                X_test.astype(self.dtype), 
                                                                y_train.astype(self.dtype), 
                                                                y_valid.astype(self.dtype), 
                                                                y_test.astype(self.dtype))
        
        return (X_train, X_valid, X_test, y_train, y_valid, y_test)

    def _get_synthetic(self):
        return None

    def _sample(self, mnist_1_X, mnist_2_X, speech_X, mnist_1_y, mnist_2_y, speech_y):
        """
        A helper function to sample from a given data split
        """
        def sample_y(y):
            mnist_1_X_y = mnist_1_X[mnist_1_y==y]
            mnist_2_X_y = mnist_2_X[mnist_2_y==y]
            speech_X_y = speech_X[speech_y==y]
            
            # get maximum possible size so that no modality is required to be repeated while sampling
            size = mnist_1_X_y.shape[0] if mnist_1_X_y.shape[0] > mnist_2_X_y.shape[0] else mnist_2_X_y.shape[0]
            # Set replacement strategy
            if mnist_1_X_y.shape[0] > mnist_2_X_y.shape[0]: replace_1, replace_2 = (False, True) 
            elif mnist_1_X_y.shape[0] < mnist_2_X_y.shape[0]: replace_1, replace_2 = (True, False) 
            elif mnist_1_X_y.shape[0] == mnist_2_X_y.shape[0]: replace_1, replace_2 = (False, False) 

            mnist_1_index = np.random.choice(mnist_1_X_y.shape[0], size=size, replace=replace_1)
            mnist_2_index = np.random.choice(mnist_2_X_y.shape[0], size=size, replace=replace_2)
            speech_index = np.random.choice(speech_X_y.shape[0], size=size, replace=True)

            # mnist 1 modality
            mnist_1_X_y = mnist_1_X_y[mnist_1_index]
            # mnist 2 modality
            mnist_2_X_y = mnist_2_X_y[mnist_2_index]
            # speech modality
            speech_X_y = speech_X_y[speech_index]
            # label Y onehots
            label_Y = np.array([self._get_onehot(y, num_digits)] * size)
            
            return (mnist_1_X_y, mnist_2_X_y, speech_X_y, label_Y)


        mnist_1 = np.zeros((0, 28, 28, 1))
        mnist_2 = np.zeros((0, 28, 28, 1))
        speech = np.zeros((0, 13))
        label_y = np.zeros((0, 10))
        for label in range(num_digits):
            mnist_1_x, mnist_2_x, speech_x, y = sample_y(label)
            mnist_1 = np.append(mnist_1, mnist_1_x, axis=0)
            mnist_2 = np.append(mnist_2, mnist_2_x, axis=0)
            speech = np.append(speech, speech_x, axis=0)
            label_y = np.append(label_y, y, axis=0)

        indices = np.arange(label_y.shape[0])
        np.random.shuffle(indices)
        
        for (m1, m2, s, digit_y) in zip(mnist_1[indices], mnist_2[indices], speech[indices], label_y[indices]):
          yield ((m1, m2, s), digit_y)
    
    def _get_onehot(self, y, num_class):
        return np.eye(num_class)[y]