import keras
import numpy as np
import pandas as pd
from copy import deepcopy
from datetime import datetime

from keras.models import Sequential
from keras.layers import Dense, LSTM, GRU, Dropout, SimpleRNN
import numpy as np
import math

from pandas.tseries.offsets import DateOffset
from pandas.tseries.offsets import *
from pandas.tseries.offsets import weekday

def dummy_predictor(csv_file):
    _df_factor_return = pd.read_csv(csv_file, sep =',',index_col=0)
    _df_factor_return.index = _df_factor_return.index.map(lambda idx: datetime.strptime(idx,"%Y-%m-%d"))
    return _df_factor_return

class RNN:
    
    def __init__(self, look_back, type='lstm', num_internal_projection=4,dropout_probability = 0.2, init ='he_uniform', loss='mse', optimizer='rmsprop'):
        self.rnn = Sequential()
        self.look_back = look_back
        if type == 'lstm':
            self.rnn.add(LSTM(num_internal_projection, batch_input_shape=(None,1,look_back), init=init))
        elif type == 'gru':
            self.rnn.add(GRU(num_internal_projection , batch_input_shape=(None,1,look_back), init=init))
        elif type == 'vanilla':
            self.rnn.add(SimpleRNN(num_internal_projection , batch_input_shape=(None,1,look_back), init=init))
        else:
            raise ValueError('Not implemented yet')
        self.rnn.add(Dropout(dropout_probability))
        self.rnn.add(Dense(1, init=init))
        self.rnn.compile(loss=loss, optimizer=optimizer, )
    
    def train(self, X, Y, nb_epoch=10, batch_size = 1):
        self.rnn.fit(X, Y, nb_epoch=nb_epoch, batch_size=batch_size, verbose=2)
    
    def evaluate(self, X, Y, batch_size = 10):
        score = self.rnn.evaluate(X, Y, batch_size = batch_size, verbose=0)
        return score
    
    def predict(self, X):
        return self.rnn.predict(X)

def series_to_matricise(dataset, look_back=10, over_lapping=True):
    n = len(dataset)
    dataX, dataY = [], []
    dataset = dataset.reshape(n)
    i = 0
    while (i + look_back) < n:
        a = dataset[i:(i+look_back)]
        dataX.append(a)
        dataY.append(dataset[i + look_back])
        if over_lapping:
            i = i + 1
        else:
            i = i + look_back + 1
    return np.array(dataX), np.array(dataY), dataset[-look_back:].reshape(1, look_back)


def rnn_predictor(factor_series, look_back=10, rnn=None, trained=False,
        type='gru', num_internal_projection=4,dropout_probability = 0.2, init ='he_uniform', loss='mse', optimizer='rmsprop',
        nb_epoch=10, batch_size = 1):
    
    factor_series = factor_series.reshape(factor_series.shape[0])
    X_train, Y_train, X_predict = series_to_matricise(factor_series, look_back, True)
    X_train = X_train.reshape([X_train.shape[0], 1, X_train.shape[1]])
    X_predict = X_predict.reshape([X_predict.shape[0], 1, X_predict.shape[1]])
    #fit rnn here
    
    if rnn is None:
        rnn = RNN(look_back, type=type, num_internal_projection=num_internal_projection, dropout_probability=dropout_probability,
              init=init, loss=loss, optimizer=optimizer)
    
    if not trained:
        rnn.train(X_train, Y_train, nb_epoch=nb_epoch, batch_size=batch_size)
    
    return rnn.predict(X_predict)[0][0]


def factor_return_rnn_predictor_daily(df_factor_return_, window_size=180,
                                look_back=10, trained=False,
                                type='gru', num_internal_projection=4,dropout_probability = 0.2, init ='he_uniform', loss='mse', optimizer='rmsprop',
                                nb_epoch=10, batch_size = 1, save_to_csv=None):
    #not recommended: slow speed
    df_factor_return = deepcopy(df_factor_return_).sort_index()
    func = lambda factor_series : rnn_predictor(factor_series, look_back, rnn=None, trained=trained,
                                      type=type, num_internal_projection=num_internal_projection,
                                      dropout_probability=dropout_probability,
                                      init=init, loss=loss, optimizer=optimizer, nb_epoch=nb_epoch,
                                      batch_size=batch_size)

    for factor in df_factor_return.columns:
        df_factor_return[factor] = pd.rolling_apply(df_factor_return[factor], window_size, func)

    if save_to_csv is not None:
        df_factor_return.dropna(axis=0).to_csv(save_to_csv)

    return df_factor_return.dropna(axis=0)


def factor_debugger(df_factor_return_, look_back=10):
    
    df_factor_return = deepcopy(df_factor_return_)
    func = lambda factor_series :  factor_series[-1]
        
    for factor in df_factor_return.columns:
            df_factor_return[factor] = pd.rolling_apply(df_factor_return[factor], look_back, func)
    
    return df_factor_return.dropna(axis=0)


def factor_return_rnn_predictor(df_factor_return_, start_date=datetime(2006,5,1),
                                        look_back=10, rnn=None,
                                        type='gru', num_internal_projection=4,dropout_probability = 0.2,
                                        init ='he_uniform', loss='mse', optimizer='rmsprop',
                                        nb_epoch=20, batch_size = 10, save_to_csv=None,
                                        train_freq="Monthly", train_period=None, verbosity=False):
    if train_freq == "Monthly":
        offset_begin = MonthBegin()
        offset_end = MonthEnd()
    else:
        raise ValueError('Frequency not implemented yet')

    df_factor_return = deepcopy(df_factor_return_).sort_index()
    predict_start = start_date + DateOffset(days=0)
    predict_end = predict_start + offset_end
    last_day = df_factor_return.index[-1]
    predict_df_list = []
    if rnn is None:
        rnn = {}
        for factor in df_factor_return.columns:
            rnn[factor] = RNN(look_back, type=type, num_internal_projection=num_internal_projection,
                  dropout_probability=dropout_probability,
                  init=init, loss=loss, optimizer=optimizer)

    while predict_start < last_day:
        print(predict_start)
        
        if train_period is None:
            df_train = df_factor_return[df_factor_return.index<predict_start]
        else:
            df_train = df_factor_return[(df_factor_return.index<predict_start)&
                                        (df_factor_return.index>=(predict_start-DateOffset(days=train_period)))]
            if verbosity:
                print("train from")
                print(df_train.index[0])
                print("to")
                print(df_train.index[-1])

        df_predict = pd.concat([
                                df_train.ix[-(look_back-1):],
                               df_factor_return[(df_factor_return.index>=predict_start)&
                                      (df_factor_return.index<=predict_end)]
                                ]).sort_index()
        
        df_res = pd.DataFrame(index=df_predict.index)

        for factor in df_predict.columns:
            factor_series = df_train[[factor]].as_matrix()
            X_train, Y_train, X_predict = series_to_matricise(factor_series, look_back, True)
            X_train = X_train.reshape([X_train.shape[0], 1, X_train.shape[1]])
            rnn[factor].train(X_train, Y_train, nb_epoch=nb_epoch, batch_size=batch_size)
            rnn_func = lambda factor_series: rnn[factor].predict(factor_series.reshape([1, 1, look_back]))
            df_res[factor] = pd.rolling_apply(df_predict[factor], look_back, rnn_func)
        
        predict_df_list.append(df_res.dropna(axis=0))
        predict_start = predict_end + offset_begin
        predict_end = predict_start + offset_end

    return pd.concat(predict_df_list).sort_index()




def example():
     df = dummy_predictor('Data/factor_return_w_industry.csv')
#    df_gru = factor_return_rnn_predictor(deepcopy(df.shift(1).dropna(axis=0)),
#                                        start_date=datetime(2006,5,1),
#                                        look_back=10, rnn=None,
#                                        type='gru', num_internal_projection=4,dropout_probability = 0.2,
#                                        init ='he_uniform', loss='mse', optimizer='rmsprop',
#                                        nb_epoch=20, batch_size = 10, save_to_csv=None,
#                                        train_freq="Monthly", train_period=255)
#    df_gru.to_csv('prediction_gru.csv')
#    
#    df_lstm = factor_return_rnn_predictor(deepcopy(df.shift(1).dropna(axis=0)),
#                                                 start_date=datetime(2006,5,1),
#                                                 look_back=10, rnn=None,
#                                                 type='lstm', num_internal_projection=4,dropout_probability = 0.2,
#                                                 init ='he_uniform', loss='mse', optimizer='rmsprop',
#                                                 nb_epoch=20, batch_size = 10, save_to_csv=None,
#                                                 train_freq="Monthly", train_period=255)
#    df_lstm.to_csv('prediction_lstm.csv')

     df_simple_RNN = factor_return_rnn_predictor(deepcopy(df.shift(1).dropna(axis=0)),
                                      start_date=datetime(2006,5,1),
                                      look_back=10, rnn=None,
                                      type='vanilla', num_internal_projection=4,dropout_probability = 0.2,
                                      init ='he_uniform', loss='mse', optimizer='rmsprop',
                                      nb_epoch=20, batch_size = 10, save_to_csv=None,
                                      train_freq="Monthly", train_period=255, verbosity=False)
                                      
     df_simple_RNN.to_csv('prediction_simple_RNN.csv')


if __name__ == "__main__":
    example()


