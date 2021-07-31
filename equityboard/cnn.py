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

n_step = 200
n_step_ahead = 1
batch_size = 1


def last_friday(day: str) -> str:
    d = date.fromisoformat(day)
    shift = (d.weekday() - 4) % 7
    shift = shift + 7 if shift < 0 else shift
    return (d - timedelta(days=shift)).isoformat()


def start_day(end_day, n_days):
    return (
            date.fromisoformat(end_day)
            - timedelta(days=(n_days + n_step_ahead))
    ).isoformat()


def scalar_train_set(
        prices: pd.DataFrame, n_days: int,
        to_day: str) -> tuple:
    from_day = start_day(to_day, n_days)
    price_arr = (prices[from_day:to_day]
                 .values.astype(np.float64))

    # scale
    sc = MinMaxScaler(feature_range=(0, 1))
    price_arr_scaled = sc.fit_transform(price_arr)

    X_train = []
    y_train = []
    for i in range(n_step, n_step + 1):
        X_train.append(price_arr_scaled[i-n_step:i, :])
        # da = np.sign(price_arr_scaled[i-n_step+n_step_ahead:i+n_step_ahead, :]
        #              - price_arr_scaled[i-n_step:i, :])
        # da[da < 0] = 0
        da = price_arr_scaled[i-n_step+n_step_ahead:i+n_step_ahead, :]
        y_train.append(da)
    X_train = np.array(X_train)
    y_train = np.array(y_train)

    return X_train, y_train


def cnn_scalar(n_channels: int) -> keras.models.Model:
    inputs = keras.layers.Input(shape=(None, n_channels))
    conv_1 = keras.layers.Conv1D(
        20, kernel_size=2, padding='causal',
        activation='relu', name='conv_1',
    )(inputs)
    conv_2 = keras.layers.Conv1D(
        20, kernel_size=2, padding='causal',
        activation='relu', dilation_rate=2, name='conv_2'
    )(conv_1)
    conv_3 = keras.layers.Conv1D(
        20, kernel_size=2, padding='causal',
        activation='relu', dilation_rate=4, name='conv_3'
    )(conv_2)
    conv_4 = keras.layers.Conv1D(
        20, kernel_size=2, padding='causal',
        activation='relu', dilation_rate=8, name='conv_4'
    )(conv_3)
    outputs = keras.layers.Conv1D(
        n_channels, kernel_size=1,
        name='output'
    )(conv_4)

    return keras.models.Model(inputs=inputs, outputs=outputs)


if __name__ == '__main__':
    to_day = '2021-05-14'
    friday = last_friday(to_day)
    n_batches = 1
    n_days = n_step
    tickers = ['ETH-USD', 'BTC-USD', 'XLM-USD',
               'XRP-USD', 'ADA-USD', 'LINK-USD']
    # tickers = ['ETH-USD']
    prices = stock_prices(tickers)
    # prices = prices.rolling(window=7).mean()
    cs = np.random.normal(1., 2., size=(1, len(tickers), 3))
    ts = np.arange(len(prices)).reshape(-1, 1) / np.pi
    # prices.loc[:, :] = (
    #     np.dot(np.sin(ts / 2), cs[..., 0])
    #     + np.dot(np.sin(ts / 5), cs[..., 1])
    #     + np.dot(np.sin(ts / 20), cs[..., 2])
    # )

    X_train, y_train = scalar_train_set(prices, n_days, friday)

    n1 = X_train.shape[1]
    n_train = int(X_train.shape[1] * 0.8)
    X_test, y_test = X_train[:, (n1-n_train):, :], y_train[:, (n1-n_train):, :]
    X_train, y_train = X_train[:, :n_train, :], y_train[:, :n_train, :]

    model = cnn_scalar(len(tickers))
    print(model.summary())

    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir='./logs', histogram_freq=1, write_images=True
    )

    optimizer = Adam()
    loss = keras.losses.BinaryCrossentropy(from_logits=True)
    model.compile(optimizer=optimizer, loss='mae')
    model.fit(X_train, y_train, epochs=400, batch_size=n_batches,
              validation_data=(X_test, y_test),
              callbacks=[tensorboard_callback])

    fname = resource_filename('resources', 'cnn_scalar.h5')
    model.save(fname, save_format='h5')

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    n_col = np.sqrt(len(tickers)).astype('int')
    n_row = np.ceil(len(tickers) / n_col).astype('int')
    fig2, axes2 = plt.subplots(n_row, n_col)
    axes2 = np.array(axes2)
    for i, ax in enumerate(axes2.ravel()):
        if i < len(tickers):
            # ax.plot(X_train[-1, :, i], label=f'{tickers[i]} X')
            ax.plot(y_train[-1, :, i],
                    label=f'{tickers[i]} true')
            ax.plot(y_pred_train[-1, :, i],
                    label=f'{tickers[i]} pred')
            ax.legend()

    n_col = np.sqrt(len(tickers)).astype('int')
    n_row = np.ceil(len(tickers) / n_col).astype('int')
    fig2, axes2 = plt.subplots(n_row, n_col)
    axes2 = np.array(axes2)
    for i, ax in enumerate(axes2.ravel()):
        if i < len(tickers):
            # ax.plot(X_test[-1, :, i], label=f'{tickers[i]} X')
            ax.plot(y_test[-1, -(n1-n_train):, i], label=f'{tickers[i]} true')
            ax.plot(y_pred_test[-1, -(n1-n_train):, i],
                    label=f'{tickers[i]} pred')
            ax.legend()

    plt.show()

