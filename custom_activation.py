import tensorflow as tf
from keras import backend as K
import numpy as np
from keras.layers import Layer, Activation
import matplotlib.pyplot as plt


class RecurrentPreisachLayer(Layer):
    """
    Attempt at a custom layer that stores the previous output in self.prev_out and previous input in prev_in
    TODO: Figure out how to get rid of InaccessibleTensorError (tf.collection?)
    """
    def __init__(self, output_dim, **kwargs):
        super(RecurrentPreisachLayer, self).__init__(**kwargs)
        self.output_dim = output_dim

    def build(self, input_shape):
        self.kernel = self.add_weight(
            name='kernel',
            shape=(input_shape[1], self.output_dim),
            initializer='normal',
            trainable=True
        )
        b_init = tf.zeros_initializer()
        self.prev_out = tf.Variable(
            initial_value=b_init(shape=((1,)), dtype='float32'),
            trainable=False,
            synchronization=tf.VariableSynchronization.ON_WRITE
        )
        self.prev_in = tf.Variable(
            initial_value=b_init(shape=((1,)), dtype='float32'),
            trainable=False,
            synchronization=tf.VariableSynchronization.ON_WRITE
        )
        super(RecurrentPreisachLayer, self).build(input_shape)

    @tf.function
    def call(self, input, mask=None):
        """
        Applies a "stop operator" to a tensor and its previous output
        y(t) = e(x(t) - x(t-1) + y(t-1))
        e(z) = min(+1, max(-1, z))
        """

        """
        ones = K.zeros_like(input) + 1
        neg_ones = K.zeros_like(input) - 1

        sum = tf.math.subtract(input, tf.math.add(self.prev_in, self.prev_out))  # x(t) - x(t-1) + y(t-1)
        e = tf.math.minimum(ones, tf.math.maximum(neg_ones, sum))  # min(+1, max(-1, z))
        self.prev_out.assign(e)
        self.prev_in.assign(input)
        """
        output = list()
        print(input.get_shape())
        idx = 0
        for i in input:
            ones = K.zeros_like(i) + 1
            neg_ones = K.zeros_like(i) - 1
            sum = tf.math.subtract(i, tf.math.add(self.prev_in, self.prev_out))
            e = tf.math.minimum(ones, tf.math.maximum(neg_ones, sum))
            self.prev_out.assign(e)
            self.prev_in.assign(i)

            output.append([e])

        #output = np.asarray(output)
        output = tf.convert_to_tensor(output)
        return output


    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)


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


def stop_operator_recurrent(x, x_prev, y_prev):
    """
    Applies a "stop operator" to a single value and its previous output
    """
    sum = y_prev + x - x_prev
    e = min(1, max(-1, sum))
    return e


def stop_operator_recurrent_tensor(x, y_prev):
    """
    Applies a "stop operator" to a tensor and its previous output
    """
    if y_prev == None:
        y_prev = K.zeros_like(x)

    ones = K.zeros_like(x) + 1
    neg_ones = K.zeros_like(x) - 1

    inv_y_prev = tf.math.pow(y_prev, -1)
    sum = tf.math.add(inv_y_prev, x)
    sum = tf.math.add(sum, tf.math.pow(x, -1))
    e = K.minimum(ones, K.maximum(neg_ones, sum))
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
    size = 500
    x_range = 10
    x = np.reshape(np.vstack((np.linspace(-x_range, x_range, num=size), np.linspace(x_range, -x_range, num=size))), size*2)
    y = np.empty_like(x)
    x_prev = -1
    y_prev = -1
    for i in range(x.shape[0]):
        y[i] = stop_operator_recurrent(x[i], x_prev, y_prev)
        x_prev = x[i]
        y_prev = y[i]
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
