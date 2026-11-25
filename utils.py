"""TimeGAN utils - faithful TF2 port of Jinsung Yoon's utils.py
Original: https://github.com/jsyoon0823/TimeGAN  (utils.py)
Reference: Yoon, Jarrett, van der Schaar, NeurIPS 2019.

Helper functions: train_test_divide, extract_time, rnn_cell,
random_generator, batch_generator.
Ported from TF1 to TF2 (tf.keras GRU/LSTM cells).
"""

import numpy as np
import tensorflow as tf


def train_test_divide(data_x, data_x_hat, data_t, data_t_hat, train_rate=0.8):
    """Divide train and test data for both original and synthetic data.
    (Faithful to Yoon's utils.train_test_divide)
    """
    no = len(data_x)
    idx = np.random.permutation(no)
    train_idx = idx[:int(no * train_rate)]
    test_idx = idx[int(no * train_rate):]

    train_x = [data_x[i] for i in train_idx]
    test_x = [data_x[i] for i in test_idx]
    train_t = [data_t[i] for i in train_idx]
    test_t = [data_t[i] for i in test_idx]

    no = len(data_x_hat)
    idx = np.random.permutation(no)
    train_idx = idx[:int(no * train_rate)]
    test_idx = idx[int(no * train_rate):]

    train_x_hat = [data_x_hat[i] for i in train_idx]
    test_x_hat = [data_x_hat[i] for i in test_idx]
    train_t_hat = [data_t_hat[i] for i in train_idx]
    test_t_hat = [data_t_hat[i] for i in test_idx]

    return train_x, train_x_hat, test_x, test_x_hat, train_t, train_t_hat, test_t, test_t_hat


def extract_time(data):
    """Returns Maximum sequence length and each sequence length.
    (Faithful to Yoon's utils.extract_time)
    """
    time = list()
    max_seq_len = 0
    for i in range(len(data)):
        max_seq_len = max(max_seq_len, len(data[i][:, 0]))
        time.append(len(data[i][:, 0]))
    return time, max_seq_len


def rnn_cell(module_name, hidden_dim):
    """Basic RNN Cell - TF2 port.
    (Yoon's rnn_cell returned a TF1 cell; here we return a tf.keras cell.)
    Args:
      - module_name: 'gru', 'lstm', or 'lstmLN'
    Returns:
      - a keras RNN cell object
    """
    assert module_name in ['gru', 'lstm', 'lstmLN']
    if module_name == 'gru':
        return tf.keras.layers.GRUCell(units=hidden_dim, activation='tanh')
    elif module_name == 'lstm':
        return tf.keras.layers.LSTMCell(units=hidden_dim, activation='tanh')
    elif module_name == 'lstmLN':
        # layer-norm LSTM not built-in; fall back to standard LSTM cell
        return tf.keras.layers.LSTMCell(units=hidden_dim, activation='tanh')


def random_generator(batch_size, z_dim, T_mb, max_seq_len):
    """Random vector generation.
    (Faithful to Yoon's utils.random_generator)
    """
    Z_mb = list()
    for i in range(batch_size):
        temp = np.zeros([max_seq_len, z_dim])
        temp_Z = np.random.uniform(0., 1, [T_mb[i], z_dim])
        temp[:T_mb[i], :] = temp_Z
        Z_mb.append(temp_Z)
    return Z_mb


def batch_generator(data, time, batch_size):
    """Mini-batch generator.
    (Faithful to Yoon's utils.batch_generator)
    """
    no = len(data)
    idx = np.random.permutation(no)
    train_idx = idx[:batch_size]

    X_mb = list(data[i] for i in train_idx)
    T_mb = list(time[i] for i in train_idx)

    return X_mb, T_mb
