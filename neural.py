import numpy as np
import tensorflow as tf

from custom_activation import stop_operator_tensor, stop_operator_recurrent_tensor, \
    RecurrentPreisachLayer


def format_data(H, B):
    """
    Format into (H, B, next H), (next B)
    """
    x_train = np.array([0.0, 0.0, 0.0])
    y_train = np.array([0])
    for i in range(len(H) - 1):
        x_val = np.array([H[i], B[i], H[i + 1]])
        x_train = np.vstack((x_train, x_val))
        y_train = np.append(y_train, B[i + 1])

    return x_train, y_train


def train_and_generate_NAME_network(x_train, y_train, save_name, n_epochs=100):
    """
    Template for generating a
    """
    save_name = "models/" + save_name
    # Define model structure
    # Generate model
    # Compile model
    # Train model
    # Save model
    # Return model


def train_and_generate_feedforward_network(x_train, y_train, save_name, n_epochs=100):
    """
    Basic feedforward network with sigmoid activation function.
    Used as control to compare to other networks.
    """
    # Define Model Structure
    model = tf.keras.models.Sequential()  # Create a sequential structure
    model.add(tf.keras.layers.Dense(3))  # Input layer (3 values for now)
    model.add(tf.keras.layers.Dense(128, activation='sigmoid'))  # Hidden layer, 128 neurons with sigmoid
    model.add(tf.keras.layers.Dense(1, activation='linear'))  # output layer (next B value)
    # Compile model
    model.compile(
        optimizer='adam',
        loss='mean_squared_error',
        metrics=[
            tf.keras.metrics.MeanSquaredError()
        ]
    )
    # Train model
    model.fit(x_train, y_train, epochs=n_epochs)
    # Save model
    save_name = "models/" + save_name
    model.save(save_name)
    #model.summary()
    # Return model
    return model


def train_and_generate_recurrent_network(x_train, y_train, save_name, n_epochs=100):
    """
    Recurrent network. Uses sigmoid operator as hidden layer
    """
    # Define Model
    model = tf.keras.models.Sequential()  # Create a sequential structure

    # alternate input layer
    #model.add(tf.keras.layers.Dense(1, activation='linear'))  # Input layer (3 values for now)

    model.add(tf.keras.layers.Dense(1))  # Input layer (only input current H)
    model.add(tf.keras.layers.Dense(128, activation='sigmoid')) # First hidden layer
    model.add(tf.keras.layers.Embedding(input_dim=1000, output_dim=64))
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128)))
    model.add(tf.keras.layers.Dense(128, activation='sigmoid')) #Second hidden layer
    model.add(tf.keras.layers.Dense(1, activation='linear'))  # output layer (current B value)
    # Compile Model
    model.compile(optimizer='adam',
                  loss='mean_squared_error',
                  metrics=[
                      tf.keras.metrics.MeanSquaredError()
                  ]
                  )
    # Train model
    model.fit(x_train, y_train, epochs=n_epochs)
    # Save model
    save_name = "models/" + save_name
    model.save(save_name)
    # model.summary()
    # Return model
    return model


def train_and_generate_preisach_network(x_train, y_train, save_name, n_epochs=100):
    """
    Preisach network uses stop operator as neuron activation function for first hidden layer
    """
    model = tf.keras.models.Sequential()
    # input layer
    model.add(tf.keras.layers.Dense(1, activation='linear'))
    # stop operator layer
    model.add(tf.keras.layers.Dense(128, activation=stop_operator_tensor))
    # sigmoid layer
    model.add(tf.keras.layers.Dense(128, activation='sigmoid'))
    # output layer
    model.add(tf.keras.layers.Dense(1, activation='linear'))
    # Compile Model
    model.compile(optimizer='adam',
                  loss='mean_squared_error',
                  metrics=[
                      tf.keras.metrics.MeanSquaredError()
                  ]
                  )
    # Train model
    model.fit(x_train, y_train, epochs=n_epochs)
    # Save model
    save_name = "models/" + save_name
    model.save(save_name)
    # model.summary()
    # Return model
    return model


def train_and_generate_recurrent_preisach_network(x_train, y_train, save_name, n_epochs=100):
    """
    Preisach network uses stop operator as neuron activation function for first hidden layer
    """
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(1, activation='linear'))  # input layer
    model.add(RecurrentPreisachLayer(10, name="stop_operator_layer", use_bias=False))  # stop operator layer
    model.add(tf.keras.layers.Dense(10, activation='sigmoid'))  # sigmoid layer
    model.add(tf.keras.layers.Dense(1, activation='linear'))  # output layer
    # Compile Model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[
            tf.keras.metrics.MeanSquaredError()
        ]
    )
    # Train model
    model.fit(x_train, y_train, epochs=n_epochs)
    # Save model
    save_name = "models/" + save_name
    #model.save(save_name)
    # model.summary()
    # Return model
    return model



def compile_recurrent_preisach_network_without_training(save_name):
    """
    Same network as 'train_and_generate_recurrent_preisach_network' but no training
    """
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(1,
                                    activation='linear',
                                    kernel_initializer=tf.keras.initializers.GlorotNormal(),
                                    bias_initializer=tf.zeros_initializer()))  # input layer
    model.add(RecurrentPreisachLayer(10,
                                     name="stop_operator_layer",
                                     use_bias=False,
                                     kernel_initializer=tf.keras.initializers.GlorotNormal(),
                                     bias_initializer=tf.zeros_initializer()))  # stop operator layer
    model.add(tf.keras.layers.Dense(10,
                                    activation='sigmoid',
                                    kernel_initializer=tf.keras.initializers.GlorotNormal(),
                                    bias_initializer=tf.zeros_initializer()))  # sigmoid layer
    model.add(tf.keras.layers.Dense(1,
                                    activation='linear',
                                    kernel_initializer=tf.keras.initializers.GlorotNormal(),
                                    bias_initializer=tf.zeros_initializer()))  # output layer
    # Compile Model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[
            tf.keras.metrics.MeanSquaredError()
        ]
    )
    return model