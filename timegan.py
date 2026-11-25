"""Time-series Generative Adversarial Networks (TimeGAN) Codebase.

FAITHFUL TF2 PORT of Jinsung Yoon's original timegan.py
Original (TF1): https://github.com/jsyoon0823/TimeGAN/blob/master/timegan.py
Reference: Yoon, Jarrett, van der Schaar,
"Time-series Generative Adversarial Networks," NeurIPS 2019.

This preserves Yoon's exact architecture, variable scopes, loss formulas
(gamma=1, 100*sqrt(G_loss_S), two-moments loss), the 3-stage training,
and the check_d_loss>0.15 conditional. Only the TF1 plumbing
(Session/placeholder/MultiRNNCell/dynamic_rnn) is converted to TF2
(keras RNN layers + GradientTape).
"""

import warnings
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Yoon's GS_solver minimizes G_loss_S over generator+supervisor vars, but
# G_loss_S only depends on the supervisor, so the generator receives no
# gradient in that isolated stage (it is trained later in joint training).
# TF1 ignored this silently; suppress the equivalent TF2 warning.
warnings.filterwarnings('ignore', message='Gradients do not exist')

from utils import extract_time, rnn_cell, random_generator, batch_generator


def timegan(ori_data, parameters):
    """TimeGAN function. Use original data as training set to generate
    synthetic data (time-series).

    Args:
      - ori_data: original time-series data  (numpy array: no x seq_len x dim)
      - parameters: dict with hidden_dim, num_layer, iterations, batch_size, module
    Returns:
      - generated_data: generated time-series data
    """

    # Basic Parameters
    no, seq_len, dim = np.asarray(ori_data).shape

    # Maximum sequence length and each sequence length
    ori_time, max_seq_len = extract_time(ori_data)

    def MinMaxScaler(data):
        """Min-Max Normalizer. (Faithful to Yoon's MinMaxScaler)"""
        min_val = np.min(np.min(data, axis=0), axis=0)
        data = data - min_val
        max_val = np.max(np.max(data, axis=0), axis=0)
        norm_data = data / (max_val + 1e-7)
        return norm_data, min_val, max_val

    # Normalization
    ori_data, min_val, max_val = MinMaxScaler(ori_data)
    ori_data = ori_data.astype('float32')

    # Network Parameters
    hidden_dim = parameters['hidden_dim']
    num_layers = parameters['num_layer']
    iterations = parameters['iterations']
    batch_size = parameters['batch_size']
    module_name = parameters['module']
    z_dim = dim
    gamma = 1

    # ----------------------------------------------------------------
    # Build the 5 networks (faithful to Yoon: MultiRNNCell of num_layers
    # GRU cells, then a sigmoid fully-connected output layer).
    # In TF2: layers.RNN(stacked cells) + Dense.
    # ----------------------------------------------------------------
    def build_net(in_dim, n_layers, out_dim, out_activation, name):
        """Build a network: stacked GRU cells -> Dense, faithful to Yoon's
        MultiRNNCell + fully_connected structure."""
        inp = keras.Input(shape=(seq_len, in_dim))
        cells = [rnn_cell(module_name, hidden_dim) for _ in range(n_layers)]
        h = layers.RNN(cells, return_sequences=True)(inp)
        out = layers.Dense(out_dim, activation=out_activation)(h)
        return keras.Model(inp, out, name=name)

    # embedder: X(dim) -> H(hidden_dim), sigmoid
    embedder = build_net(dim, num_layers, hidden_dim, 'sigmoid', 'embedder')
    # recovery: H(hidden_dim) -> X_tilde(dim), sigmoid
    recovery = build_net(hidden_dim, num_layers, dim, 'sigmoid', 'recovery')
    # generator: Z(z_dim) -> E(hidden_dim), sigmoid
    generator = build_net(z_dim, num_layers, hidden_dim, 'sigmoid', 'generator')
    # supervisor: H(hidden_dim) -> S(hidden_dim), sigmoid  (num_layers-1)
    supervisor = build_net(hidden_dim, num_layers - 1, hidden_dim, 'sigmoid', 'supervisor')
    # discriminator: H(hidden_dim) -> Y_hat(1), NO activation (logits)
    discriminator = build_net(hidden_dim, num_layers, 1, None, 'discriminator')

    # Optimizers (Yoon uses default AdamOptimizer for all 5 solvers)
    E0_solver = keras.optimizers.Adam()
    E_solver = keras.optimizers.Adam()
    D_solver = keras.optimizers.Adam()
    G_solver = keras.optimizers.Adam()
    GS_solver = keras.optimizers.Adam()

    mse = keras.losses.MeanSquaredError()

    def sigmoid_ce(labels, logits):
        return tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits))

    # ----------------------------------------------------------------
    # Training step functions (faithful to Yoon's loss definitions)
    # ----------------------------------------------------------------
    @tf.function
    def train_embedder(X_mb):
        with tf.GradientTape() as tape:
            H = embedder(X_mb, training=True)
            X_tilde = recovery(H, training=True)
            E_loss_T0 = mse(X_mb, X_tilde)
            E_loss0 = 10.0 * tf.sqrt(E_loss_T0)
        vars_er = embedder.trainable_variables + recovery.trainable_variables
        grads = tape.gradient(E_loss0, vars_er)
        E0_solver.apply_gradients(zip(grads, vars_er))
        return E_loss_T0

    @tf.function
    def train_supervisor(X_mb):
        with tf.GradientTape() as tape:
            H = embedder(X_mb, training=True)
            H_hat_supervise = supervisor(H, training=True)
            G_loss_S = mse(H[:, 1:, :], H_hat_supervise[:, :-1, :])
        vars_gs = generator.trainable_variables + supervisor.trainable_variables
        grads = tape.gradient(G_loss_S, vars_gs)
        GS_solver.apply_gradients(zip(grads, vars_gs))
        return G_loss_S

    @tf.function
    def train_generator(X_mb, Z_mb):
        with tf.GradientTape() as tape:
            H = embedder(X_mb, training=True)
            E_hat = generator(Z_mb, training=True)
            H_hat = supervisor(E_hat, training=True)
            H_hat_supervise = supervisor(H, training=True)
            X_hat = recovery(H_hat, training=True)
            Y_fake = discriminator(H_hat, training=True)
            Y_fake_e = discriminator(E_hat, training=True)

            # 1. Adversarial loss
            G_loss_U = sigmoid_ce(tf.ones_like(Y_fake), Y_fake)
            G_loss_U_e = sigmoid_ce(tf.ones_like(Y_fake_e), Y_fake_e)
            # 2. Supervised loss
            G_loss_S = mse(H[:, 1:, :], H_hat_supervise[:, :-1, :])
            # 3. Two Moments
            mean_xhat, var_xhat = tf.nn.moments(X_hat, axes=[0])
            mean_x, var_x = tf.nn.moments(X_mb, axes=[0])
            G_loss_V1 = tf.reduce_mean(tf.abs(
                tf.sqrt(var_xhat + 1e-6) - tf.sqrt(var_x + 1e-6)))
            G_loss_V2 = tf.reduce_mean(tf.abs(mean_xhat - mean_x))
            G_loss_V = G_loss_V1 + G_loss_V2
            # 4. Summation
            G_loss = (G_loss_U + gamma * G_loss_U_e
                      + 100.0 * tf.sqrt(G_loss_S) + 100.0 * G_loss_V)
        vars_gs = generator.trainable_variables + supervisor.trainable_variables
        grads = tape.gradient(G_loss, vars_gs)
        G_solver.apply_gradients(zip(grads, vars_gs))
        return G_loss_U, G_loss_S, G_loss_V

    @tf.function
    def train_embedder_joint(X_mb):
        with tf.GradientTape() as tape:
            H = embedder(X_mb, training=True)
            X_tilde = recovery(H, training=True)
            H_hat_supervise = supervisor(H, training=True)
            E_loss_T0 = mse(X_mb, X_tilde)
            E_loss0 = 10.0 * tf.sqrt(E_loss_T0)
            G_loss_S = mse(H[:, 1:, :], H_hat_supervise[:, :-1, :])
            E_loss = E_loss0 + 0.1 * G_loss_S
        vars_er = embedder.trainable_variables + recovery.trainable_variables
        grads = tape.gradient(E_loss, vars_er)
        E_solver.apply_gradients(zip(grads, vars_er))
        return E_loss_T0

    @tf.function
    def compute_d_loss(X_mb, Z_mb):
        H = embedder(X_mb, training=False)
        E_hat = generator(Z_mb, training=False)
        H_hat = supervisor(E_hat, training=False)
        Y_fake = discriminator(H_hat, training=False)
        Y_real = discriminator(H, training=False)
        Y_fake_e = discriminator(E_hat, training=False)
        D_loss_real = sigmoid_ce(tf.ones_like(Y_real), Y_real)
        D_loss_fake = sigmoid_ce(tf.zeros_like(Y_fake), Y_fake)
        D_loss_fake_e = sigmoid_ce(tf.zeros_like(Y_fake_e), Y_fake_e)
        return D_loss_real + D_loss_fake + gamma * D_loss_fake_e

    @tf.function
    def train_discriminator(X_mb, Z_mb):
        with tf.GradientTape() as tape:
            D_loss = compute_d_loss(X_mb, Z_mb)
        grads = tape.gradient(D_loss, discriminator.trainable_variables)
        D_solver.apply_gradients(zip(grads, discriminator.trainable_variables))
        return D_loss

    def get_batch():
        X_mb, T_mb = batch_generator(ori_data, ori_time, batch_size)
        return np.array(X_mb, dtype='float32'), T_mb

    def get_noise(T_mb):
        Z_mb = random_generator(batch_size, z_dim, T_mb, max_seq_len)
        return np.array(Z_mb, dtype='float32')

    # ----------------------------------------------------------------
    # 1. Embedding network training
    # ----------------------------------------------------------------
    print('Start Embedding Network Training', flush=True)
    for itt in range(iterations):
        X_mb, T_mb = get_batch()
        step_e_loss = train_embedder(X_mb)
        if itt % 1000 == 0:
            print('step: ' + str(itt) + '/' + str(iterations) +
                  ', e_loss: ' + str(np.round(np.sqrt(step_e_loss), 4)), flush=True)
    print('Finish Embedding Network Training', flush=True)

    # ----------------------------------------------------------------
    # 2. Training only with supervised loss
    # ----------------------------------------------------------------
    print('Start Training with Supervised Loss Only', flush=True)
    for itt in range(iterations):
        X_mb, T_mb = get_batch()
        step_g_loss_s = train_supervisor(X_mb)
        if itt % 1000 == 0:
            print('step: ' + str(itt) + '/' + str(iterations) +
                  ', s_loss: ' + str(np.round(np.sqrt(step_g_loss_s), 4)), flush=True)
    print('Finish Training with Supervised Loss Only', flush=True)

    # ----------------------------------------------------------------
    # 3. Joint Training
    # ----------------------------------------------------------------
    print('Start Joint Training', flush=True)
    step_d_loss = 0.0
    for itt in range(iterations):
        # Generator training (twice more than discriminator)
        for kk in range(2):
            X_mb, T_mb = get_batch()
            Z_mb = get_noise(T_mb)
            step_g_loss_u, step_g_loss_s, step_g_loss_v = train_generator(X_mb, Z_mb)
            X_mb, T_mb = get_batch()
            step_e_loss_t0 = train_embedder_joint(X_mb)

        # Discriminator training
        X_mb, T_mb = get_batch()
        Z_mb = get_noise(T_mb)
        check_d_loss = float(compute_d_loss(X_mb, Z_mb))
        if check_d_loss > 0.15:
            step_d_loss = float(train_discriminator(X_mb, Z_mb))

        if itt % 1000 == 0:
            print('step: ' + str(itt) + '/' + str(iterations) +
                  ', d_loss: ' + str(np.round(step_d_loss, 4)) +
                  ', g_loss_u: ' + str(np.round(float(step_g_loss_u), 4)) +
                  ', g_loss_s: ' + str(np.round(np.sqrt(float(step_g_loss_s)), 4)) +
                  ', g_loss_v: ' + str(np.round(float(step_g_loss_v), 4)) +
                  ', e_loss_t0: ' + str(np.round(np.sqrt(float(step_e_loss_t0)), 4)), flush=True)
    print('Finish Joint Training', flush=True)

    # ----------------------------------------------------------------
    # Synthetic data generation
    # ----------------------------------------------------------------
    Z_mb = random_generator(no, z_dim, ori_time, max_seq_len)
    Z_mb = np.array(Z_mb, dtype='float32')
    E_hat = generator(Z_mb, training=False)
    H_hat = supervisor(E_hat, training=False)
    generated_data_curr = recovery(H_hat, training=False).numpy()

    generated_data = list()
    for i in range(no):
        temp = generated_data_curr[i, :ori_time[i], :]
        generated_data.append(temp)

    # Renormalization
    generated_data = generated_data * max_val
    generated_data = generated_data + min_val

    return generated_data
