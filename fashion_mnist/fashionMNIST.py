from __future__ import absolute_import, division, print_function, unicode_literals
import importlib
import pandas as pd
import numpy as np
import utils
from sklearn.model_selection import train_test_split

import tensorflow as tf
layers, models = tf.keras.layers, tf.keras.models

importlib.reload(utils)

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


def get_data():
    fashion_mnist = tf.keras.datasets.fashion_mnist

    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    # Combine train and test images and labels
    images = np.concatenate((train_images, test_images))
    labels = np.concatenate((train_labels, test_labels))
    # Flatten images to 1D arrays
    images = images.reshape(-1, 28 * 28)
    # Convert images and labels to DataFrame
    data = {'image_' + str(i): images[:, i] for i in range(images.shape[1])}
    data['label'] = labels
    df = pd.DataFrame(data)
    df_shuffled = df.sample(frac=1).reset_index(drop=True)
    return df_shuffled


def split_data0(df_shuffled):
    X = df_shuffled.drop('label', axis=1)
    y = df_shuffled['label']
    X_train_, X_test, y_train_, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    X_train, X_validation, y_train, y_validation = train_test_split(X_train_, y_train_, test_size=0.1, random_state=42)

    # To transform the images to be in scale from 0-1
    X_train = X_train / 255.0
    X_test = X_test / 255.0
    X_validation = X_validation / 255.0
    return X_train, y_train, X_test, y_test, X_validation, y_validation


def split_data1(df_shuffled):
    X = df_shuffled.drop('label', axis=1)
    y = df_shuffled['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

    # To transform the images to be in scale from 0-1
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    return X_train, y_train, X_test, y_test

