# Temporary main function for one model
import polars as pl
from configs import *
from helpers import *
from KerasModels import model_class


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
learning_rate = 3e-3 # found from learning rate
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

plot_series(valid_time, (valid_set, forecast), title = "Forecast_vs_realized_hybrid_lstm")
plot_series(valid_time, (valid_set, forecast), start = -30, title = "Forecast_vs_realized_zoom_hybrid_lstm")
