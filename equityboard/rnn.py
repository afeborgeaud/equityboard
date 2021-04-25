import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from datetime import date, timedelta
from tensorflow.keras.layers import Dense, Input, LSTM
from tensorflow.keras.layers import TimeDistributed, Conv1D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import tensorflow.keras as keras
from capm import stock_prices
import matplotlib.pyplot as plt
from pkg_resources import resource_filename

n_step = 300
n_step_ahead = 56
batch_size = 32


def last_friday(day: str) -> str:
    d = date.fromisoformat(day)
    shift = (d.weekday() - 4) % 7
    shift = shift + 7 if shift < 0 else shift
    return (d - timedelta(days=shift)).isoformat()


def start_day(end_day, n_days):
    return (
            date.fromisoformat(end_day)
            - timedelta(days=(n_days + n_step + n_step_ahead))
    ).isoformat()


def scalar_train_set(
        prices: pd.Series, n_12weeks: int,
        to_day: str) -> tuple:
    n_day = n_12weeks * batch_size
    from_day = start_day(to_day, n_12weeks)
    price_arr = (prices[from_day:to_day]
                 .values.astype(np.float64))

    # scale
    sc = MinMaxScaler(feature_range=(0, 1))
    price_arr_scaled = sc.fit_transform(price_arr)

    X_train = []
    y_train = []
    for i in range(n_step, n_day + n_step):
        X_train.append(price_arr_scaled[i-n_step:i, 0])
        y_train.append(price_arr_scaled[i:i+n_step_ahead, 0])
    X_train = (np.array(X_train)
               .reshape((n_day, n_step, 1)))
    y_train = (np.array(y_train)
               .reshape((n_day, n_step_ahead, 1)))
    return X_train, y_train, sc


def vector_train_set(
        prices: pd.Series, n_12weeks: int,
        to_day: str) -> tuple:
    n_day = n_12weeks * batch_size
    from_day = start_day(to_day, n_day)
    price_arr = (prices[from_day:to_day]
                 .values.astype(np.float64))

    # scale
    sc = MinMaxScaler(feature_range=(0, 1))
    price_arr_scaled = sc.fit_transform(price_arr)

    X_train = np.empty((n_day, n_step, 1))
    y_train = np.empty((n_day, n_step, n_step_ahead))
    for i in range(0, n_day):
        X_train[i, :] = price_arr_scaled[i:i + n_step]
        for step_ahead in range(1, n_step_ahead + 1):
            y_train[i, :, step_ahead-1] = (
                price_arr_scaled[
                    i+step_ahead:i+step_ahead+n_step, 0
                ]
            )

    return X_train, y_train, sc


def train_dataset(
        prices: pd.Series, n_12weeks: int,
        to_day: str) -> tuple:

    n_day = n_12weeks * batch_size
    from_day = start_day(to_day, n_12weeks)
    price_arr = (prices[from_day:to_day]
                 .values.astype(np.float64))

    # scale
    sc = MinMaxScaler(feature_range=(0, 1))
    price_arr_scaled = sc.fit_transform(price_arr)

    dataset = tf.data.Dataset.from_tensor_slices(price_arr_scaled)
    window_length = n_step + n_step_ahead
    dataset = dataset.window(window_length, shift=1,
                             drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(window_length))
    dataset = dataset.batch(batch_size)

    dataset = dataset.map(lambda windows:
                          (windows[:, :-n_step_ahead],
                           windows[:, -n_step_ahead:]))
    dataset = dataset.prefetch(1)

    return dataset, sc


def lstm_vector() -> keras.models.Model:
    inputs = keras.layers.Input(shape=(None, 1))
    lstm_1 = keras.layers.LSTM(20, return_sequences=True)(inputs)
    lstm_2 = keras.layers.LSTM(20, return_sequences=True)(lstm_1)
    outputs = keras.layers.TimeDistributed(
        Dense(n_step_ahead))(lstm_2)

    return keras.models.Model(inputs=inputs, outputs=outputs)


def lstm_vector_conv() -> keras.models.Model:
    inputs = keras.layers.Input(shape=(None, 1))
    conv_1 = keras.layers.Conv1D(filters=20, kernel_size=4, strides=2,
                    padding='valid')(inputs)
    lstm_1 = keras.layers.LSTM(
        20, return_sequences=True,
        dropout=0.2,
        recurrent_dropout=0.2
    )(conv_1)
    lstm_2 = keras.layers.LSTM(
        20, return_sequences=True,
        dropout=0.2,
        recurrent_dropout=0.2
    )(lstm_1)
    outputs = keras.layers.TimeDistributed(
        Dense(n_step_ahead)
    )(lstm_2)

    return keras.models.Model(inputs=inputs, outputs=outputs)


if __name__ == '__main__':
    to_day = '2021-04-15'
    friday = last_friday(to_day)
    n_12weeks = 6
    from_day = start_day(friday, n_12weeks)
    prices = stock_prices(['SPY'])
    prices = prices.rolling(window=7).mean()
    prices.loc[:, 'SPY'] = (
        np.sin(np.arange(len(prices)) / (2*np.pi))
        + np.sin(np.arange(len(prices)) / (5*np.pi))
        + np.sin(np.arange(len(prices)) / (20*np.pi))
    )

    X_train, y_train, sc = vector_train_set(prices, n_12weeks, friday)
    y_train = y_train[:, 3::2, :]
    model = lstm_vector_conv()
    print(model.summary())

    optimizer = Adam()
    model.compile(optimizer=optimizer, loss='mae')
    model.fit(X_train, y_train, epochs=200, batch_size=batch_size)

    fname = resource_filename('resources', 'rnn_lstm.h5')
    model.save(fname, save_format='h5')

    X_test = prices.values[-n_step-n_step_ahead:-n_step_ahead]
    y_test = prices.values[-n_step_ahead:]
    X_test_scaled = sc.transform(X_test)
    X_test_scaled = X_test_scaled.reshape(1, -1, 1)
    y_test_scaled = sc.transform(y_test)
    y_hat = model.predict(X_test_scaled)

    y_test = y_test[3::2]

    price_test = np.hstack((X_test_scaled[0, :, 0], y_test_scaled[:, 0]))
    price_preds = [np.hstack((X_test_scaled[0, :, 0], y_hat[0, i, :]))
                   for i in range(y_hat.shape[1])]

    plt.plot(price_test, label='test')
    plt.plot(price_preds[-1], label='pred')
    plt.legend()
    plt.show()

