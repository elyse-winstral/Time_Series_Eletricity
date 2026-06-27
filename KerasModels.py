from pathlib import Path
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from configs import *
from helpers import plot_series
# from beartype import beartype

# @beartype
class model_class:
    
    def __init__(self, model: tf.keras.models, 
                 train_set: tf.data.Dataset | None, 
                 n_epochs_lr: int | None, 
                 n_epochs_train: int | None,
                 name: str,
                 ):
        
        self.model = model
        self.train_set = train_set
        self.n_epochs_lr = n_epochs_lr
        self.n_epochs_train = n_epochs_train
        self.name = name
        self.init_weights = None
        self.window_size = window_size
        self.batch_size = batch_size


    def print_structure(self): 
        print(f"Model Summary: {self.name}")
        print(self.model.summary())

    
    @property
    def get_learning_rate(self):
            init_weights = self.model.get_weights()

            starting_rate = 1e-5

            lr_schedule = tf.keras.callbacks.LearningRateScheduler(
                lambda epoch: starting_rate * 10 ** (epoch / 20)
            )

            optimizer = tf.keras.optimizers.Adam()
            loss = tf.keras.losses.Huber()
            self.model.compile(loss = loss, optimizer = optimizer)
            history = self.model.fit(self.train_set, epochs = self.n_epochs_lr, callbacks = [lr_schedule])

            lrs = starting_rate * 10 ** (np.arange(self.n_epochs_lr)/20)
            plt.figure(figsize=(10,6))
            plt.grid(True)
            plt.semilogx(lrs, history.history["loss"])
            plt.ylim((0, 0.2))
            plt.tick_params("both", length = 10, width = 1, which = "both")
            plt.savefig(FIG_DIR / f"Loss_rate_{self.name}.png")
            plt.close()
            self.init_weights = init_weights
            return 
    
    #@property
    def set_learning_rate(self, value: float):
        self.learning_rate = value

    @property
    def train(self):
        assert(self.learning_rate is not None)
        tf.keras.backend.clear_session()
        if self.init_weights is not None:
            self.model.set_weights(self.init_weights)
        optimizer = tf.keras.optimizers.Adam(learning_rate = self.learning_rate)
        loss = tf.keras.losses.Huber()
        self.model.compile(loss = loss, optimizer = optimizer, metrics = ["mae"])
        history = self.model.fit(self.train_set, epochs = self.n_epochs_train)

        mae = history.history["mae"]
        loss = history.history["loss"]

        epochs = range(len(loss))

        plot_series(x = epochs, y = (mae, loss), title=f"Entire_MAE_and_Loss_{self.name}", xlabel = 'Epochs', legend=['MAE', "Loss"])
        
        zoom_split = int(epochs[-1]*0.2)
        epochs_zoom = epochs[zoom_split:]
        mae_zoom = mae[zoom_split:]
        loss_zoom = loss[zoom_split:]

        plot_series(x = epochs_zoom, y = (mae_zoom, loss_zoom), title=f"Zoomed_MAE_and_loss_{self.name}", xlabel = 'Epochs', legend=['MAE', "Loss"])


    def forecast(self, series):
        series = tf.expand_dims(series, axis = -1)

        dataset = tf.data.Dataset.from_tensor_slices(series)
        dataset = dataset.window(self.window_size, shift = 1, drop_remainder=True)
        dataset = dataset.flat_map(lambda w: w.batch(self.batch_size))
        dataset = dataset.batch(self.batch_size).prefetch(1)
        forecast = self.model.predict(dataset, verbose = 0)
        return forecast
    
        
    
    @property
    def save(self, name: str | None = None):
        if name is not None:
            self.model.save(MODEL_DIR / f"{name}.keras")
        else:
            self.model.save(MODEL_DIR / f"{self.name}.keras")


if __name__ == "__main__":
    import polars as pl
    from helpers import *

    data_path = DATA_DIR / "100233.csv"

    df = pl.read_csv(data_path,separator=";")
    df = df.with_columns(pl.col("Start der Messung").str.slice(0,10).str.to_date().alias("Datum"))
    df = df.group_by("Datum").agg(pl.col("Stromverbrauch").sum())
    df = df.with_columns(((pl.col("Stromverbrauch") - pl.col("Stromverbrauch").mean() )/ pl.col("Stromverbrauch").std()).alias("Scaled"))

    df = df.sort(by = "Datum")

    series = df.select(["Scaled"]).to_numpy().flatten()
    time = np.arange(len(series))

    # scale values 
    train_set = series[:split_time]
    train_time = time[:split_time]
    valid_set = series[split_time:]
    valid_time = time[split_time:]

    training_dataset = windowed_dataset(train_set, window_size, batch_size, shuffle_buffer_size)
    valid_dataset = windowed_dataset(valid_set, window_size, batch_size, shuffle_buffer_size)

    relu_kernel_initializer = tf.keras.initializers.HeUniform()
    lam = 0.001 #[1e-4, 1e-2]
    l2 = tf.keras.regularizers.L2(lam)

    hybrid_lstm_structure = tf.keras.models.Sequential([
        tf.keras.Input(shape=(window_size,1)),
        tf.keras.layers.Conv1D(filters=31, kernel_size = 7, strides = 1, padding = "causal", activation = "relu", kernel_regularizer = l2),
        tf.keras.layers.LSTM(31, return_sequences = False, kernel_regularizer = l2, recurrent_regularizer=l2),
        tf.keras.layers.Dense(30, activation= "relu", kernel_initializer= relu_kernel_initializer, kernel_regularizer = l2,),
        tf.keras.layers.Dense(1, activation='linear'),
    ])

    hybrid_lstm = model_class(hybrid_lstm_structure, train_set = training_dataset, n_epochs_lr = 100, n_epochs_train= 500, name = "hybrid_lstm")
    print(hybrid_lstm.name)

    hybrid_lstm.print_structure()
    # hybrid_lstm.get_learning_rate
    learning_rate = 3e-3
    hybrid_lstm.learning_rate = learning_rate
    hybrid_lstm.train
    hybrid_lstm.save
    train_forecast = hybrid_lstm.forecast(train_set[:-1]).squeeze()

    print("Training Results")
    print(f"MSE: {tf.keras.metrics.mse(train_set[window_size:], train_forecast).numpy()}")
    print(f"MAE: {tf.keras.metrics.mae(train_set[window_size:], train_forecast).numpy()}")

    forecast = hybrid_lstm.forecast(series[split_time - window_size: -1]).squeeze()
    print("Test Results")
    print(f"MSE: {tf.keras.metrics.mse(valid_set, forecast).numpy()}")
    print(f"MAE: {tf.keras.metrics.mae(valid_set, forecast).numpy()}")

    plot_series(valid_time, (valid_set, forecast), title = "Forecast_vs_realized")
    plot_series(valid_time, (valid_set, forecast), start = -30, title = "Forecast_vs_realized_zoom")
