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
    """
    Applies a "stop operator" function to a single value
    """
    e = min(1, max(-1, x))
    return e


def stop_operator_tensor(x):
    """
    Applies a "stop operator" function to a tensor
    """
    # array of positive and negative ones
    ones = K.zeros_like(x) + 1
    neg_ones = K.zeros_like(x) - 1
    e = K.minimum(ones, K.maximum(neg_ones, x))
    return e


def stop_operator_recurrent(x, y_prev):
    """
    Applies a "stop operator" to a single value and its previous output
    """
    sum = y_prev**-1 + x + x**-1
    e = min(1, max(-1, sum))
    return e

class StopOperator:
    def __init__(self, y_prev):
        self.y_prev = y_prev
    def stop_operator_recurrent_tensor(self, x):
        """
        Applies a "stop operator" to a tensor and its previous output
        """
        if y_prev == None:
            y_prev = K.zeros_like(x)

        ones = K.zeros_like(x) + 1
        neg_ones = K.zeros_like(x) - 1

        sum = y_prev**-1 + x + x**-1
        e = K.minimum(ones, K.maximum(neg_ones, sum))
        y_prev = e
        return e


def make_activator(activations):
    """
    Creates an activation function that applies 2 functions to a tensor through slicing
    """
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


# =================================================================================================
# =========================================== PLOTTING ============================================
# =================================================================================================

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


def plot_recurrent_activation_function():
    size = 20
    x = np.reshape(np.vstack((np.linspace(-1, 1, num=size), np.linspace(1, -1, num=size))), size*2)
    y = np.empty_like(x)
    x_prev = -2
    for i in range(x.shape[0]):
        y[i] = stop_operator_recurrent(x[i], x_prev)
        x_prev = x[i]
    plt.plot(x, y)
    plt.show()


# =================================================================================================
# ====================================== EXECUTABLE SCRIPT ========================================
# =================================================================================================


# Test stop operator
# plot_activation_function(stop_operator)
# Compare stop operator activation function to other activation functions
# plot_activation_function(K.sigmoid)
# plot_activation_function(K.relu)
# plot_tensor_activation_function(K.softmax)
# plot_tensor_activation_function(stop_operator_tensor)

plot_recurrent_activation_function()
