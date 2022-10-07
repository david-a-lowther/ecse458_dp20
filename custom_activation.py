import tensorflow as tf
from tensorflow import keras
from keras import backend as K
import numpy as np
import csv
from keras.utils.generic_utils import get_custom_objects
import tensorflow.keras.metrics as metrics
from keras.layers import Layer, Activation
import itertools


class LayerTest(Layer):
    def __init__(self, num_units, activation):
        super(LayerTest, self).__init__()
        self.num_units = num_units
        self.activation = Activation(activation)

    def build(self, input_shape):
        self.weight = self.add_weight(shape=[input_shape[-1], self.num_units])
        self.bias = self.add_weight(shape=[self.num_units])

    def call(self, input, mask=None):
        y = tf.matmul(input, self.weight) + self.bias
        y = self.activation(y)
        return y

    # model.add(LayerTest(128, activation='relu'))


def stop_operator(x):
    return K.sigmoid(x)


def make_activator(activations):
    def activator(t):
        slices = tf.unstack(t, num=32, axis=0)
        activated = []
        i = 0
        for slice in slices:
            if i < len(slices) - 1:
                activated.append(activations[0](slice))
            else:
                activated.append(activations[1](slice))
            i += 1
        return tf.stack(activated)
    return activator


def train_neural_net(x_train, y_train, n_epochs=100):
    model = tf.keras.models.Sequential()  # Create a sequential structure
    model.add(tf.keras.layers.Dense(1, activation='linear'))  # Input layer (3 values for now)
    model.add((tf.keras.layers.Dense(128, activation=make_activator([K.sigmoid, K.sigmoid])))) # Hidden layer, 128 neurons with sigmoid
    model.add(tf.keras.layers.Dense(128, activation='sigmoid'))
    model.add(tf.keras.layers.Dense(1, activation='linear'))  # output layer (next B value)
    model.compile(
        optimizer='adam',
        loss='mean_squared_error',
        metrics=[
            metrics.MeanSquaredError()
        ]
    )

    model.fit(x_train, y_train, epochs=n_epochs)
    model.save('models/custom_activation_test.model')
    model.summary()


def extract_csv_info(filename) -> []:
    """Extracts most crucial data_simulated from csv file (for the purpose of this project)"""
    file = open('data_simulated/' + filename)
    csvreader = csv.reader(file)
    H = np.array([])
    B = np.array([])
    i = 0
    for row in csvreader:
        if i < 7:
            i += 1
            continue
        H = np.append(H, float(row[2]))
        B = np.append(B, float(row[3]))
    file.close()
    H = H[0:1984]
    B = B[0:1984]
    return H[:, np.newaxis], B[:, np.newaxis]


d = extract_csv_info('20PNF1500 - Sheet1.csv')
train_neural_net(d[0], d[1])