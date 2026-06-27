from pathlib import Path
import tensorflow as tf
import matplotlib.pyplot as plt

# ====================================
# Configuation
# ====================================


MAIN_DIR = Path(__file__).parent
FIG_DIR = MAIN_DIR / "figures"
MODEL_DIR = MAIN_DIR / "trained_models"
DATA_DIR = MAIN_DIR / "data"

FIG_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

# roughly 80 - 20 split
split_time = 4000 

# LSTM
window_size = 21
batch_size = 32
shuffle_buffer_size = 1000
