import tensorflow as tf
from tensorflow import keras
from keras import backend as K
import numpy as np
import csv
from keras.utils.generic_utils import get_custom_objects
import tensorflow.keras.metrics as metrics
from keras.layers import Layer, Activation
import itertools
import matplotlib.pyplot as plt


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
    e = min(1, max(-1, x))
    return e


# applies operator to a tensor instead of individual values like above
def stop_operator_tensor(x):
    # array of positive and negative ones
    ones = K.zeros_like(x) + 1
    neg_ones = K.zeros_like(x) - 1
    e = K.minimum(ones, K.maximum(neg_ones, x))
    return e


def plot_activation_function(activation_f):
    x = np.linspace(-2, 2, num=100)
    y = np.empty_like(x)
    for i in range(x.shape[0]):
        y[i] = activation_f(x[i])
    plt.plot(x, y)
    plt.show()


# different function because K.softmax takes input as tensor
# instead of individual values, so above function will not work for it
def plot_tensor_activation_function(activation_tensor_f):
    x = tf.linspace(-2, 2, num=100)
    y = activation_tensor_f(x)
    plt.plot(x, y)
    plt.show()


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
    model.add((tf.keras.layers.Dense(128, activation=make_activator(
        [K.sigmoid, K.sigmoid]))))  # Hidden layer, 128 neurons with sigmoid
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


def train_preisach_neural_net(x_train, y_train, n_epochs=100):
    tf.keras.backend.set_floatx('float64')
    model = tf.keras.models.Sequential()
    # input layer
    model.add(tf.keras.layers.Dense(1, activation='linear'))
    # stop operator layer
    model.add(tf.keras.layers.Dense(128, activation=stop_operator_tensor))
    # sigmoid layer
    model.add(tf.keras.layers.Dense(128, activation='sigmoid'))
    # output layrer
    model.add(tf.keras.layers.Dense(1, activation='linear'))

    model.compile(
        optimizer='adam',
        loss='mean_squared_error',
        metrics=[
            metrics.MeanSquaredError()
        ]
    )
    # train
    model.fit(x_train, y_train, epochs=n_epochs)
    model.save('models/pnn.model')
    model.summary()
    return model


# Given actual output and predicted output
# predict MSE (mean squared error)
def compute_mse(actual_y, predicted_y):
    diff = np.subtract(actual_y, predicted_y)
    squared = np.square(diff)
    mse = np.mean(squared)
    return mse


# given list of tuples (x, y)
# shuffle and split into training and testing set
# This function needs a lot of work, quick and dirty for now
def shuffle_and_split(d, info=False):
    d = np.asarray(d)
    # need to shuffle but keep x and y together

    x = d[0]
    y = d[1]
    x = np.asarray(x)
    y = np.asarray(y)

    if info:
        print(str(d[0][1]) + ", " + str(d[1][1]))
        print(str(d[0][2]) + ", " + str(d[1][2]))
        print(str(d[0][3]) + ", " + str(d[1][3]))
        print(str(d[0][4]) + ", " + str(d[1][4]))
        print(str(d[0][0]) + ", " + str(d[1][0]))
        print(str(d[0][10]) + ", " + str(d[1][10]))

    train_x = list()
    train_y = list()
    test_x = list()
    test_y = list()

    # for now just put every 10th element into test set
    for i in range(len(x)):
        if i % 10 == 0:
            test_x.append(d[0][i])
            test_y.append(d[1][i])
        else:
            train_x.append(d[0][i])
            train_y.append(d[1][i])

    train = np.column_stack((train_x, train_y))
    test = np.column_stack((test_x, test_y))

    print(train[0].shape)
    train = np.swapaxes(train, 0, 1)
    test = np.swapaxes(test, 0, 1)
    print(train[0].shape)

    # train_x = tf.convert_to_tensor(train_x)
    # train_y = tf.convert_to_tensor(train_y)
    # test_x = tf.convert_to_tensor(test_x)
    # test_y = tf.convert_to_tensor(test_y)

    if info:
        train_x = train[0]
        train_y = train[1]
        test_x = test[0]
        test_y = test[1]
        print("After")
        print(str(train_x[0]) + ", " + str(train_y[0]))
        print(str(train_x[1]) + ", " + str(train_y[1]))
        print(str(train_x[2]) + ", " + str(train_y[2]))
        print(str(train_x[3]) + ", " + str(train_y[3]))

        print(str(test_x[0]) + ", " + str(test_y[0]))
        print(str(test_x[1]) + ", " + str(test_y[1]))

    return train, test


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


# test stop operator
#plot_activation_function(stop_operator)
#compare stop operator activation function to other activation functions
#plot_activation_function(K.sigmoid)
#plot_activation_function(K.relu)
#plot_tensor_activation_function(K.softmax)
#plot_tensor_activation_function(stop_operator_tensor)


#d = extract_csv_info('20PNF1500 - Sheet1.csv')
#model = train_neural_net(d[0], d[1])


# train and test priesach neural network
d = extract_csv_info('20PNF1500 - Sheet1.csv')
train, test = shuffle_and_split(d, True)

# train
model = train_preisach_neural_net(train[0], train[1], 10)

# test
prediction = model.predict(test[0])
# compute mean squared error
mse = compute_mse(test[1], prediction)
# compare first 10 elements of actual vs predicated values for test output
print(test[0][:10])
print(test[1][:10])
print(list(prediction[:10]))
print("MSE (test set): " + str(mse))
