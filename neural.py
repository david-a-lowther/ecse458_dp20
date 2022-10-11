import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras.metrics as metrics


def format_data(H, B):
    """
    Format into (H, B, next H), (next B)
    """
    x_train = np.array([0.0,0.0,0.0])
    y_train = np.array([0])
    for i in range(len(H) - 1):
        x_val = np.array([H[i], B[i], H[i+1]])
        x_train = np.vstack((x_train, x_val))
        y_train = np.append(y_train, B[i+1])

    return x_train, y_train

def train_and_generate_NAME_network(x_train, y_train, savename, n_epochs=100):
    savename = "models/" + savename
    # Define model structure
    # Generate model
    # Compile model
    # Train model
    # Save model
    # Return model



def train_and_generate_network_feedforward(x_train, y_train, n_epochs):
    model = tf.keras.models.Sequential()  # Create a sequential structure
    model.add(tf.keras.layers.Dense(3))  # Input layer (3 values for now)
    model.add(tf.keras.layers.Dense(128, activation='sigmoid')) # Hidden layer, 128 neurons with sigmoid
    model.add(tf.keras.layers.Dense(1, activation='linear'))  # output layer (next B value)
    model.compile(
        optimizer='adam',
        loss='mean_squared_error',
        metrics=[
            metrics.MeanSquaredError()
        ]
    )

    model.fit(x_train, y_train, epochs=n_epochs)
    model.save('models/feedforward_preliminary.model')
    model.summary()


def train_and_generate_network_recurrent(in_train, out_train, n_epochs):
    # TODO: Not functional yet
    model = tf.keras.models.Sequential()  # Create a sequential structure
    model.add(tf.keras.layers.Dense(3))  # Input layer (3 values for now)
    model.add(tf.keras.layers.LSTM(128))  # First attempt at recurrent layers
    model.add(tf.keras.layers.Dense(1, activation='linear'))  # output layer (next B value)

    model.compile(optimizer='adam',
                  loss='mean_squared_error',
                  metrics=[
                      metrics.MeanSquaredError()
                  ]
    )

    model.fit(in_train, out_train, epochs=n_epochs)
    model.save('models/recurrent_1.model')
    model.summary()


# =========================================================================
# ============================ TESTING ====================================
# =========================================================================


def extract_csv_info(filename) -> []:
    """Extracts most crucial data_simulated from csv file (for the purpose of this project)"""
    file = open('datasets/' + filename)
    csvreader = csv.reader(file)
    H = []
    B = []
    i = 0
    for row in csvreader:
        H.append(float(row[0]))
        B.append(float(row[1]))
    file.close()
    return H, B


def test_gen_net_ff(n):
    info = extract_csv_info("HB.csv")
    train_data = format_data(info[0], info[1])
    train_and_generate_network_feedforward(train_data[0], train_data[1], n_epochs=n)


def test_gen_net_rnn(n):
    info = extract_csv_info("HB.csv")
    train_data = format_data(info[0], info[1])
    train_and_generate_network_recurrent(train_data[0], train_data[1], n_epochs=n)


if __name__ == '__main__':
    test_gen_net_ff(100)
    # test_gen_net_rnn(100)

    # model = tf.keras.models.load_model('models/feedforward_preliminary.model')
    # inp = np.array([[7.17912892092092,-0.0621395139889629, 7.46961185285285]])
    # prediction = model.predict(inp)
    # print("==================TEST WITH ONE VALUE==================")
    # print("Actual value is -0.056295319419508, prediction by network:", prediction[0,0])
