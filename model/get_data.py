from warnings import catch_warnings

import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
import cv2
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import OneHotEncoder

# creating instance of one-hot-encoder

class LoadData:

    def __init__(self):
        # column order in CSV file
        column_names = ['id', 'path', 'age', 'categorical']

        # bring in the data...
        train_dataset = pd.read_csv(filepath_or_buffer="database_ageMio.csv",
                                    names=column_names, header=0)

        column_features = train_dataset.values[:, 1]
        column_labels = train_dataset.values[:, 3]

        # split up the training data and define as features/labels
        # column_features = train_dataset.drop(columns=train_dataset.columns[[0, 2]], axis=1)
        # column_labels = train_dataset.drop(columns=train_dataset.columns[[0, 1]], axis=1)

        # array_features = column_features.to_numpy()

        # array_features = array_features

        features = np.array(list(map(map_fn, column_features)))
        labels = column_labels
        
        self.train_X, self.test_X, self.train_y, self.test_y = train_test_split(features,
                                                                                labels,
                                                                                test_size=0.10,
                                                                                random_state=42)

        self.train_X, self.validation_X, self.train_y, self.validation_y = train_test_split(self.train_X,
                                                                                            self.train_y,
                                                                                            test_size=0.20,
                                                                                        random_state=13)
        self.train_y = to_categorical(self.train_y)
        self.test_y = to_categorical(self.test_y)
        self.validation_y = to_categorical(self.validation_y)
		
		"""
        # self.train_y = (self.train_y == "menores") * 1
        # self.test_y = (self.test_y == "menores") * 1
        # self.validation_y = (self.validation_y == "menores") * 1
        # self.red()
        
        #self.test_y = to_categorical(labels)
        #self.test_X = features"""
        

def red(self):
    index = np.where(self.train_y == 1)

    for i in range(len(self.train_X)):
        if i not in index:
            np.delete(self.train_X, i)
            np.delete(self.train_y, i)


def expand(label):
    return tf.expand_dims(label, -1)


def map_fn(path):
    image = cv2.imread(("../../" + path))
    return cv2.resize(image, (224,224), interpolation=cv2.INTER_AREA)
    
