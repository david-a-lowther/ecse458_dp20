import tensorflow as tf
from tensorflow import keras
from keras import backend as K
import numpy as np
from keras.utils.generic_utils import get_custom_objects
import tensorflow.keras.metrics as metrics
from keras.layers import Layer, Activation
import matplotlib.pyplot as plt

from data_preprocessing import *


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


def train_neural_net(x_train, y_train, n_epochs=100, recurrent=False, savename="control.model"):
    model = tf.keras.models.Sequential()  # Create a sequential structure
    model.add(tf.keras.layers.Dense(1, activation='linear'))  # Input layer (3 values for now)

    # model.add((tf.keras.layers.Dense(128, activation=make_activator(
    #     [K.sigmoid, K.sigmoid]))))  # Hidden layer, 128 neurons with sigmoid
    model.add(tf.keras.layers.Dense(128, activation='sigmoid'))

    if recurrent:
        model.add(tf.keras.layers.Embedding(input_dim=1000, output_dim=64))
        model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128)))

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
    model.save('models/' + str(savename))
    model.summary()
    return model


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
    actual_y = np.asarray(actual_y)
    predicted_y = np.asarray(predicted_y)
    diff = np.subtract(actual_y, predicted_y)
    squared = np.square(diff)
    mse = np.mean(squared)
    return mse


# test compute mse
y = [11,20,19,17,10]
y_pred = [12,18,19.5,18,9]
print(compute_mse(y, y_pred))


# test stop operator
#plot_activation_function(stop_operator)
#compare stop operator activation function to other activation functions
#plot_activation_function(K.sigmoid)
#plot_activation_function(K.relu)
#plot_tensor_activation_function(K.softmax)
#plot_tensor_activation_function(stop_operator_tensor)


#d = extract_csv_info('20PNF1500 - Sheet1.csv')
#model = train_neural_net(d[0], d[1])


# train and test control vs preisach neural network
raw_data = extract_csv_info("./data_simulated/M19_29Gauge - Sheet1.csv")
# format into (current H, current B, next H, next B)
formatted_data = format_data(raw_data)
train, test = shuffle_and_split(formatted_data)
train_x, train_y = split_input_output(train)
test_x, test_y = split_input_output(test)


# train normal net
print("Control NN:")
control_model = train_neural_net(train_x, train_y, 20)
# test pnn
control_pred = control_model.predict(test_x)
# compute error
control_mse = compute_mse(test_y, control_pred)
print("MSE: " + str(control_mse))
# compare first 10 elements
print("Actual:")
print(test_y[:10])
print("Predicted:")
print(control_pred.tolist()[:10])

print("Evaluate Control NN:")
control_model.evaluate(test_x, test_y)


print("-------------------------------------")


# load model
# training recurrent is slow so train once and then load model
#rnn_model = train_neural_net(train_x, train_y, 20, recurrent=True, savename="recurrent_test2.model")
rnn1_model = tf.keras.models.load_model("./models/recurrent_test.model")
rnn2_model = tf.keras.models.load_model("./models/recurrent_test2.model")


print("-------------------------------------")

print("Preisach NN:")
# train pnn
pnn_model = train_preisach_neural_net(train_x, train_y, 20)
# test pnn
pnn_pred = pnn_model.predict(test_x)
# compute error
pnn_mse = compute_mse(test_y, pnn_pred)
print("MSE: " + str(pnn_mse))
# compare first 10 elements
print("Actual:")
print(test_y[:10])
print("Predicted:")
print(pnn_pred.tolist()[:10])


print("Evaluate Preisach NN:")
pnn_model.evaluate(test_x, test_y)


# graph pnn output
test_data = extract_csv_info("./data_simulated/M19_TESTINGDATA - M19_TESTINGDATA.csv")
formatted_test_data = format_data(test_data)
f_test_data_x, f_test_data_y = split_input_output(formatted_test_data)

next_H = list()
for i in range(len(f_test_data_x)):
    next_H.append(f_test_data_x[i][2])

# plot actual data
plt.figure(figsize=(20, 12))
plt.xlim(-750, 750)
plt.plot(next_H, f_test_data_y, marker="o", color='black')
plt.title("Actual Data Plot")
plt.xlabel("Magnetic Field H (T)")
plt.ylabel("Magnetic Flux B (A/m)")
plt.show()

#plot control net predicted data
control_pred_next_b = control_model.predict(f_test_data_x)
# plot actual data
plt.figure(figsize=(20, 12))
plt.xlim(-750, 750)
plt.plot(next_H, control_pred_next_b, marker="o", color='black')
plt.title("Control Net predicted Plot")
plt.xlabel("Magnetic Field H (T)")
plt.ylabel("Magnetic Flux B (A/m)")
plt.show()

# plot test recurrent nn predicted data
rnn1_pred_next_b = rnn1_model.predict(f_test_data_x)
# plot actual data
plt.figure(figsize=(20, 12))
plt.xlim(-750, 750)
plt.plot(next_H, rnn1_pred_next_b, marker="o", color='black')
plt.title("Recurrent NN predicted Plot")
plt.xlabel("Magnetic Field H (T)")
plt.ylabel("Magnetic Flux B (A/m)")
plt.show()

# plot test recurrent nn predicted data
rnn2_pred_next_b = rnn2_model.predict(f_test_data_x)
# plot actual data
plt.figure(figsize=(20, 12))
plt.xlim(-750, 750)
plt.plot(next_H, rnn2_pred_next_b, marker="o", color='black')
plt.title("Recurrent NN predicted Plot")
plt.xlabel("Magnetic Field H (T)")
plt.ylabel("Magnetic Flux B (A/m)")
plt.show()

# plot pnn predicted data
pnn_pred_next_b = pnn_model.predict(f_test_data_x)
# plot actual data
plt.figure(figsize=(20, 12))
plt.xlim(-750, 750)
plt.plot(next_H, pnn_pred_next_b, marker="o", color='black')
plt.title("PNN predicted Plot")
plt.xlabel("Magnetic Field H (T)")
plt.ylabel("Magnetic Flux B (A/m)")
plt.show()