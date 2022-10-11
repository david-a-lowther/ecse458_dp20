import tensorflow as tf
from keras import backend as K
import numpy as np
from keras.layers import Layer, Activation
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

# test stop operator
# plot_activation_function(stop_operator)
# compare stop operator activation function to other activation functions
# plot_activation_function(K.sigmoid)
# plot_activation_function(K.relu)
# plot_tensor_activation_function(K.softmax)
# plot_tensor_activation_function(stop_operator_tensor)
