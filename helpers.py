import matplotlib.pyplot as plt
import tensorflow as tf
from configs import *


def plot_series(x, y, format = "-", start = 0, end = None, title = None, xlabel=None, ylabel = None, legend=None):
    plt.figure(figsize=(10,6))

    if type(y) is tuple:
        for y_curr in y:
            plt.plot(x[start:end], y_curr[start: end], format)
    else:
        plt.plot(x[start:end], y[start:end], format)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if legend:
        plt.legend(legend)
    plt.title(title)
    plt.grid(True)
    plt.savefig(FIG_DIR / f"{title}.png")
    plt.close()

def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
    series = tf.expand_dims(series, axis = -1)

    dataset = tf.data.Dataset.from_tensor_slices(series)
    dataset = dataset.window(window_size + 1, shift = 1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(window_size+1))
    dataset = dataset.map(lambda window: (window[:-1], window[-1]))
    dataset = dataset.shuffle(shuffle_buffer)
    dataset = dataset.batch(batch_size)
    dataset = dataset.cache().prefetch(1)

    return dataset


