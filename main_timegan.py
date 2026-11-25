"""Main TimeGAN runner - faithful to Yoon's main_timegan.py structure,
adapted to load your household energy data (cons.csv) and save outputs.

Original: https://github.com/jsyoon0823/TimeGAN/blob/master/main_timegan.py
Reference: Yoon, Jarrett, van der Schaar, NeurIPS 2019.

Run:  apptainer exec --nv timegan.sif python main_timegan.py
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf

from timegan import timegan

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
for gpu in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)
np.random.seed(42)
tf.random.set_seed(42)

# ====================================================================
# PATHS
# ====================================================================
DATA_PATH = '../data/cons.csv'
OUTPUT_PATH = 'output/'
os.makedirs(OUTPUT_PATH + 'visualizations', exist_ok=True)
os.makedirs(OUTPUT_PATH + 'synthetic_data', exist_ok=True)

# ====================================================================
# PARAMETERS  (Yoon's defaults; edit for testing vs real run)
#   Yoon's example: seq_len 24, module gru, hidden_dim 24,
#                   num_layer 3, iteration 50000, batch_size 128
# ====================================================================
SEQ_LEN   = 168      # weekly window (you can use 24 for daily like Yoon)
MAX_SEQ   = 5000     # how many sequences to use (cap for speed)

parameters = {
    'module':     'gru',
    'hidden_dim': 24,
    'num_layer':  3,
    'iterations': 10000,   # <-- SMOKE TEST: set to 10; REAL: 10000-50000
    'batch_size': 128,
}

# ====================================================================
# LOAD + PREP DATA  (build sequences shaped no x seq_len x 1)
# ====================================================================
print("Loading data...", flush=True)
raw = pd.read_csv(DATA_PATH, header=None).values.astype('float32')
raw = np.clip(pd.DataFrame(raw).ffill(axis=1).bfill(axis=1).values, 0, None)
print(f"Raw shape: {raw.shape}", flush=True)

seqs = []
for h in range(raw.shape[0]):
    row = raw[h]
    for s in range(0, len(row) - SEQ_LEN + 1, SEQ_LEN):
        seqs.append(row[s:s + SEQ_LEN].reshape(-1, 1))
ori_data = np.array(seqs, dtype='float32')

# subsample
if len(ori_data) > MAX_SEQ:
    idx = np.random.choice(len(ori_data), MAX_SEQ, replace=False)
    ori_data = ori_data[idx]
print(f"Training sequences: {ori_data.shape}", flush=True)

# ====================================================================
# RUN TIMEGAN  (Yoon's function)
# ====================================================================
print("="*60, flush=True)
print("RUNNING TIMEGAN (faithful port of Yoon's code)", flush=True)
print("="*60, flush=True)

generated_data = timegan(ori_data, parameters)
generated_data = np.array(generated_data)
print(f"Generated shape: {generated_data.shape}", flush=True)

# ====================================================================
# SAVE
# ====================================================================
np.save(OUTPUT_PATH + 'synthetic_data/synthetic_timegan.npy', generated_data)
# also save a flat 2D csv (sequences x seq_len)
flat = generated_data.reshape(generated_data.shape[0], -1)
pd.DataFrame(flat).to_csv(
    OUTPUT_PATH + 'synthetic_data/synthetic_timegan.csv', index=False, header=False)
# save the real sequences used (for evaluation)
np.save(OUTPUT_PATH + 'synthetic_data/ori_data_used.npy', ori_data)
print("Saved synthetic data.", flush=True)

# ====================================================================
# QUICK COMPARISON PLOT
# ====================================================================
fig, ax = plt.subplots(3, 2, figsize=(14, 8))
for i in range(3):
    ri = np.random.randint(0, len(ori_data))
    si = np.random.randint(0, len(generated_data))
    ax[i, 0].plot(ori_data[ri].reshape(-1))
    ax[i, 0].set_title(f'Original - sample {i+1}')
    ax[i, 1].plot(generated_data[si].reshape(-1), color='crimson')
    ax[i, 1].set_title(f'Synthetic - sample {i+1}')
    ax[i, 0].grid(alpha=0.3); ax[i, 1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_PATH + 'visualizations/comparison.png', dpi=100, bbox_inches='tight')
plt.close()
print("Saved comparison plot.", flush=True)

print("="*60, flush=True)
print("DONE. Outputs in", OUTPUT_PATH, flush=True)
print("="*60, flush=True)
