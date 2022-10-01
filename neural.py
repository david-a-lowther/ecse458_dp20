import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras.metrics as metrics


def format_data(data_in, data_out):
    # TODO:
    # Format into (H, B), (next H, next B)
    pass


def train_and_generate_network_feedforward(in_train, out_train, n_epochs):
    # TODO: Not functional yet
    model = tf.keras.models.Sequential()  # Create a sequential structure
    model.add(tf.keras.layers.Flatten(3, input_shape=(1, 3)))  # Input layer (3 values for now)
    model.add(tf.keras.layers.Dense(1024, activation='sigmoid'))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))  # output layer (next B value)

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=[
            metrics.Accuracy(),
            metrics.MeanSquaredError()
        ]
    )


def train_and_generate_network_recurrent(in_train, out_train, n_epochs):
    # TODO: Not functional yet
    model = tf.keras.models.Sequential()  # Create a sequential structure
    model.add(tf.keras.layers.Flatten(3, input_shape=(1, 3)))  # Input layer (3 values for now)
    model.add(tf.keras.layers.LSTM(1024))  # RNN Layer, first attempt, need to refine
    model.add(tf.keras.layers.LSTM(1024))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))  # output layer (next B value)

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=[
                      metrics.Accuracy(),
                      metrics.MeanSquaredError()
                  ]
    )

    model.fit(in_train, out_train, epochs=n_epochs)
    model.save('recurrent_1.model')


def run_net():
    print(1)


train_and_generate_network_recurrent(
    [(0.0, 0.0, 0.0),(1.0, 1.0, 1.0),(2.0, 2.0, 2.0),(3.0, 3.0, 3.0),(4.0, 4.0, 4.0)],
    [0.0, 1.0, 2.0, 3.0, 4.0],
    1
)
run_net()