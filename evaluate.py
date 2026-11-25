"""
TimeGAN Evaluation - the 3 paper metrics (TensorFlow 2)
Adapted from Jinsung Yoon's official TimeGAN repo:
  https://github.com/jsyoon0823/TimeGAN  (metrics/ directory)
Reference: Yoon, Jarrett, van der Schaar, NeurIPS 2019.

Metrics:
  1. Discriminative score - can a GRU classifier tell real from fake?
     output = |accuracy - 0.5|   (lower = better, 0 = perfect)
  2. Predictive score (TSTR) - train GRU on synthetic, test on real.
     output = MAE   (lower = better; compare to real-trained benchmark)
  3. Visualization - PCA + t-SNE of real vs synthetic.

Run:  apptainer exec --nv timegan.sif python evaluate.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
np.random.seed(42)
tf.random.set_seed(42)

OUTPUT_PATH = 'output/'
DATA_PATH = '../data/cons.csv'
SEQ_LEN = 168
N_EVAL = 1000          # how many sequences to sample for evaluation

# ====================================================================
# LOAD REAL + SYNTHETIC, build comparable weekly sequences
# ====================================================================
print("Loading data...", flush=True)
real_full = pd.read_csv(DATA_PATH, header=None).values.astype('float32')
real_full = np.clip(pd.DataFrame(real_full).ffill(axis=1).bfill(axis=1).values, 0, None)
synth_full = np.load(OUTPUT_PATH + 'synthetic_data/synthetic_timegan.npy')

def to_sequences(arr):
    seqs = []
    for h in range(arr.shape[0]):
        row = arr[h]
        for s in range(0, len(row) - SEQ_LEN + 1, SEQ_LEN):
            seqs.append(row[s:s+SEQ_LEN].reshape(-1, 1))
    return np.array(seqs, dtype='float32')

real_seq = to_sequences(real_full)
synth_seq = to_sequences(synth_full)
print(f"Real sequences:  {real_seq.shape}", flush=True)
print(f"Synth sequences: {synth_seq.shape}", flush=True)

# normalise both to [0,1] together (so the classifier judges shape, not scale)
both_max = max(real_seq.max(), synth_seq.max())
both_min = min(real_seq.min(), synth_seq.min())
real_seq = (real_seq - both_min) / (both_max - both_min + 1e-7)
synth_seq = (synth_seq - both_min) / (both_max - both_min + 1e-7)

# sample equal numbers
n = min(N_EVAL, len(real_seq), len(synth_seq))
ri = np.random.choice(len(real_seq), n, replace=False)
si = np.random.choice(len(synth_seq), n, replace=False)
ori_data = real_seq[ri]
gen_data = synth_seq[si]
dim = ori_data.shape[2]

# ====================================================================
# METRIC 1: DISCRIMINATIVE SCORE
# (GRU classifier; output |accuracy - 0.5|)
# ====================================================================
def discriminative_score(ori, gen):
    print("\n[1] Discriminative score...", flush=True)
    hidden = max(int(dim/2), 8)
    # build classifier: GRU -> last state -> dense(1)
    model = keras.Sequential([
        keras.Input(shape=(ori.shape[1], dim)),
        layers.GRU(hidden, activation='tanh'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # label real=1, fake=0; split 80/20
    X = np.concatenate([ori, gen], axis=0)
    y = np.concatenate([np.ones(len(ori)), np.zeros(len(gen))])
    perm = np.random.permutation(len(X))
    X, y = X[perm], y[perm]
    split = int(0.8 * len(X))
    Xtr, Xte = X[:split], X[split:]
    ytr, yte = y[:split], y[split:]

    model.fit(Xtr, ytr, epochs=20, batch_size=128, verbose=0)
    pred = (model.predict(Xte, verbose=0).ravel() > 0.5).astype(int)
    acc = accuracy_score(yte, pred)
    score = abs(0.5 - acc)
    print(f"    classifier accuracy = {acc:.4f}", flush=True)
    print(f"    DISCRIMINATIVE SCORE = {score:.4f}  (lower better, 0=perfect)", flush=True)
    return score

# ====================================================================
# METRIC 2: PREDICTIVE SCORE (TSTR)
# train on one set to predict next step, test MAE on real
# ====================================================================
def predictive_model(train, test):
    # predict next-step value from past; input = seq[:-1], target = seq[1:]
    hidden = max(int(dim/2), 8)
    model = keras.Sequential([
        keras.Input(shape=(train.shape[1]-1, dim)),
        layers.GRU(hidden, activation='tanh', return_sequences=True),
        layers.Dense(dim)
    ])
    model.compile(optimizer='adam', loss='mae')
    Xtr, Ytr = train[:, :-1, :], train[:, 1:, :]
    model.fit(Xtr, Ytr, epochs=20, batch_size=128, verbose=0)
    Xte, Yte = test[:, :-1, :], test[:, 1:, :]
    pred = model.predict(Xte, verbose=0)
    return mean_absolute_error(Yte.reshape(-1), pred.reshape(-1))

def predictive_score(ori, gen):
    print("\n[2] Predictive score (TSTR)...", flush=True)
    # train on synthetic, test on real
    synth_mae = predictive_model(gen, ori)
    # benchmark: train on real, test on real (split)
    half = len(ori)//2
    real_mae = predictive_model(ori[:half], ori[half:])
    print(f"    PREDICTIVE (trained on synthetic, tested on real) = {synth_mae:.4f}", flush=True)
    print(f"    PREDICTIVE (trained on real, benchmark)           = {real_mae:.4f}", flush=True)
    print(f"    -> closer together = better", flush=True)
    return synth_mae, real_mae

# ====================================================================
# METRIC 3: VISUALIZATION (PCA + t-SNE)
# ====================================================================
def visualization(ori, gen):
    print("\n[3] PCA + t-SNE visualization...", flush=True)
    # flatten sequences to vectors
    o = ori.reshape(len(ori), -1)
    g = gen.reshape(len(gen), -1)
    m = min(len(o), len(g), 500)
    o, g = o[:m], g[:m]
    combined = np.concatenate([o, g], axis=0)

    pca = PCA(n_components=2).fit_transform(combined)
    tsne = TSNE(n_components=2, perplexity=40, n_iter=300,
                random_state=42).fit_transform(combined)

    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    ax[0].scatter(pca[:m, 0], pca[:m, 1], c='blue', alpha=0.4, s=20, label='Original')
    ax[0].scatter(pca[m:, 0], pca[m:, 1], c='red', alpha=0.4, s=20, label='Synthetic')
    ax[0].set_title('PCA: Original vs Synthetic'); ax[0].legend(); ax[0].grid(alpha=0.3)
    ax[1].scatter(tsne[:m, 0], tsne[:m, 1], c='blue', alpha=0.4, s=20, label='Original')
    ax[1].scatter(tsne[m:, 0], tsne[m:, 1], c='red', alpha=0.4, s=20, label='Synthetic')
    ax[1].set_title('t-SNE: Original vs Synthetic'); ax[1].legend(); ax[1].grid(alpha=0.3)
    plt.tight_layout()
    out = OUTPUT_PATH + 'visualizations/pca_tsne_evaluation.png'
    plt.savefig(out, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"    saved: {out}", flush=True)

# ====================================================================
# RUN ALL THREE
# ====================================================================
print("\n" + "="*60, flush=True)
print("EVALUATION", flush=True)
print("="*60, flush=True)

d_score = discriminative_score(ori_data, gen_data)
p_synth, p_real = predictive_score(ori_data, gen_data)
visualization(ori_data, gen_data)

print("\n" + "="*60, flush=True)
print("SUMMARY", flush=True)
print("="*60, flush=True)
print(f"Discriminative score : {d_score:.4f}   (0=perfect, 0.5=worst)", flush=True)
print(f"Predictive (synthetic): {p_synth:.4f}", flush=True)
print(f"Predictive (real/benchmark): {p_real:.4f}", flush=True)
print(f"Visualization: output/visualizations/pca_tsne_evaluation.png", flush=True)
print("="*60, flush=True)
