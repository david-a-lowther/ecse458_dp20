import keras.layers
import tensorflow as tf
from keras import backend as K
import numpy as np
from keras.layers import Layer, Activation
import matplotlib.pyplot as plt
from keras.dtensor import utils
from keras.engine.input_spec import InputSpec
from keras.engine.base_layer import Layer


class RecurrentPreisachLayer(keras.layers.Dense):
    def build(self, input_shape, **kwargs):
        dtype = tf.as_dtype(self.dtype or backend.floatx())
        if not (dtype.is_floating or dtype.is_complex):
            raise TypeError(
                "A Dense layer can only be built with a floating-point "
                f"dtype. Received: dtype={dtype}"
            )

        input_shape = tf.TensorShape(input_shape)
        last_dim = tf.compat.dimension_value(input_shape[-1])
        if last_dim is None:
            raise ValueError(
                "The last dimension of the inputs to a Dense layer "
                "should be defined. Found None. "
                f"Full input shape received: {input_shape}"
            )
        self.input_spec = InputSpec(min_ndim=2, axes={-1: last_dim})
        self.kernel = self.add_weight(
            "kernel",
            shape=[last_dim, self.units],
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            dtype=self.dtype,
            trainable=True,
        )
        if self.use_bias:
            self.bias = self.add_weight(
                "bias",
                shape=[
                    self.units,
                ],
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                dtype=self.dtype,
                trainable=True,
            )
        else:
            self.bias = None
        self.built = True
        b_init = tf.zeros_initializer()
        self.prev_out = tf.Variable(
            initial_value=b_init(shape=((10,)), dtype='float32'),
            trainable=False,
            synchronization=tf.VariableSynchronization.ON_WRITE
        )
        self.prev_in = tf.Variable(
            initial_value=b_init(shape=((10,)), dtype='float32'),
            trainable=False,
            synchronization=tf.VariableSynchronization.ON_WRITE
        )

    def call(self, inputs):
        if inputs.dtype.base_dtype != self._compute_dtype_object.base_dtype:
            inputs = tf.cast(inputs, dtype=self._compute_dtype_object)

        is_ragged = isinstance(inputs, tf.RaggedTensor)
        if is_ragged:
            # In case we encounter a RaggedTensor with a fixed last dimension
            # (last dimension not ragged), we can flatten the input and restore
            # the ragged dimensions at the end.
            if tf.compat.dimension_value(inputs.shape[-1]) is None:
                raise ValueError(
                    "Dense layer only supports RaggedTensors when the "
                    "innermost dimension is non-ragged. Received: "
                    f"inputs.shape={inputs.shape}."
                )
            original_inputs = inputs
            if inputs.flat_values.shape.rank > 1:
                inputs = inputs.flat_values
            else:
                # Innermost partition is encoded using uniform_row_length.
                # (This is unusual, but we can handle it.)
                if inputs.shape.rank == 2:
                    inputs = inputs.to_tensor()
                    is_ragged = False
                else:
                    for _ in range(original_inputs.ragged_rank - 1):
                        inputs = inputs.values
                    inputs = inputs.to_tensor()
                    original_inputs = tf.RaggedTensor.from_nested_row_splits(
                        inputs, original_inputs.nested_row_splits[:-1]
                    )

        rank = inputs.shape.rank
        if rank == 2 or rank is None:
            # We use embedding_lookup_sparse as a more efficient matmul
            # operation for large sparse input tensors. The op will result in a
            # sparse gradient, as opposed to
            # sparse_ops.sparse_tensor_dense_matmul which results in dense
            # gradients. This can lead to sigfinicant speedups, see b/171762937.
            if isinstance(inputs, tf.SparseTensor):
                # We need to fill empty rows, as the op assumes at least one id
                # per row.
                inputs, _ = tf.sparse.fill_empty_rows(inputs, 0)
                # We need to do some munging of our input to use the embedding
                # lookup as a matrix multiply. We split our input matrix into
                # separate ids and weights tensors. The values of the ids tensor
                # should be the column indices of our input matrix and the
                # values of the weights tensor can continue to the actual matrix
                # weights.  The column arrangement of ids and weights will be
                # summed over and does not matter. See the documentation for
                # sparse_ops.sparse_tensor_dense_matmul a more detailed
                # explanation of the inputs to both ops.
                ids = tf.SparseTensor(
                    indices=inputs.indices,
                    values=inputs.indices[:, 1],
                    dense_shape=inputs.dense_shape,
                )
                weights = inputs
                outputs = tf.nn.embedding_lookup_sparse(
                    self.kernel, ids, weights, combiner="sum"
                )
            else:
                outputs = tf.matmul(a=inputs, b=self.kernel)
        # Broadcast kernel to inputs.
        else:
            outputs = tf.tensordot(inputs, self.kernel, [[rank - 1], [0]])
            # Reshape the output back to the original ndim of the input.
            if not tf.executing_eagerly():
                shape = inputs.shape.as_list()
                output_shape = shape[:-1] + [self.kernel.shape[-1]]
                outputs.set_shape(output_shape)

        if self.use_bias:
            outputs = tf.nn.bias_add(outputs, self.bias)

        if True:  # Modified portion of the layer
            """
            Applies a "stop operator" to a tensor and its previous output
            y(t) = e(x(t) - x(t-1) + y(t-1))
            e(z) = min(+1, max(-1, z))
            """
            last_input = tf.unstack(outputs)[-1]
            input_vector = tf.unstack(outputs)
            unstacked_neuron_out = [stop_operator_tensor(  # First input
                tf.math.subtract(last_input, tf.math.add(self.prev_in, self.prev_out))
            )]
            for j in range(1, len(input_vector)):
                sum = tf.math.subtract(input_vector[j], tf.math.add(input_vector[j - 1], input_vector[j - 1]))  # x(t) - x(t-1) + y(t-1)
                e = stop_operator_tensor(sum)  # min(1, max(-1, sum))
                unstacked_neuron_out.append(e)
            # unstacked_neuron_out.append(last_input)

            self.prev_in.assign(last_input)  # Assign the last input in batch to prev_in
            self.prev_out.assign(unstacked_neuron_out[-1])  # Assign last output in batch to prev_out
            outputs = tf.stack(unstacked_neuron_out)

        if is_ragged:
            outputs = original_inputs.with_flat_values(outputs)

        return outputs


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
    e = tf.math.minimum(ones, tf.math.maximum(neg_ones, x))
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


def plot_stop_operator_af(dynamic=False, sine_input=False):
    """
    Plots the stop operator activation function in response to an input range from -x_range to x_range
    If sine_input is true, the input sequence is a sinusoid. else, it is a linearly increasing/decreasing input
    if dynamic is true, the plot is generated as an "animation"
    """

    figure, axis = plt.subplots(2, 1, figsize=(8, 8))

    size = 200  # Number of samples for input
    x_range = 5
    trail_sample_length = 20
    if sine_input:
        x = x_range * np.sin(np.linspace(0, x_range*4, num=size*2))
    else:
        x = np.reshape(np.vstack((np.linspace(-x_range, x_range, num=size), np.linspace(x_range, -x_range, num=size))), size*2)

    y = np.empty_like(x)
    x_prev = -1
    y_prev = -1

    for i in range(x.shape[0]):
        y[i] = stop_operator_recurrent(x[i], x_prev, y_prev)
        x_prev = x[i]
        y_prev = y[i]
        if dynamic:
            low_range = i-trail_sample_length
            if low_range < 0:
                low_range=0

            axis[0].clear()
            axis[0].set_title('Input Sequence')
            axis[0].set_ylim(-x_range - 0.5, x_range + 0.5)
            axis[0].set_xlim(-1, trail_sample_length)
            axis[0].plot(x[low_range:i])

            axis[1].clear()
            axis[1].set_title('Stop operator output')
            axis[1].set_ylim(-1.2, 1.2)
            axis[1].set_xlim(-x_range - 0.5, x_range + 0.5)
            axis[1].plot(x[low_range:i], y[low_range:i], color='red')

            plt.pause(0.0001)
    if not dynamic:
        axis[0].plot(x)
        axis[0].set_title('Input Sequence')
        axis[1].set_title('Stop operator output')
        axis[1].plot(x, y, color='red')
    plt.show(block=True)


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
# plot_stop_operator_af(dynamic=True, sine_input=True)
