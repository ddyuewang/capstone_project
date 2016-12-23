# -*- coding: utf-8 -*-
"""
Created on Sat Nov 19 12:19:22 2016

@author: wyx
"""

import pickle
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from copy import deepcopy
import timeit

class view_data_reader():
    
    def __init__(self, file_dir, config, error_periods=60, error_method = "rolling_window"):
        self._df_factor_return = pd.read_csv(file_dir, sep =',',index_col=0)
        self._df_factor_return.index = self._df_factor_return.index.map(lambda idx: datetime.strptime(idx,"%Y-%m-%d"))
        self._df_factor_return.sort_index(inplace=True)
        self.config = config
        self.error_periods = error_periods
        self.error_method =error_method
        self._estimation = self.get_estimation()
        if self.error_method == "rolling_window":
            self._error = self.get_standard_error(periods=self.error_periods)
        elif self.error_method == "exponential":
            self._error = self.get_exponential_decay_error(periods=self.error_periods)
        else:
            raise ValueError(error_method + " not implemented")

    def reset(self, file_dir=None, config=None, error_periods=None, error_method = None):
        if file_dir is not None:
            self._df_factor_return = pd.read_csv(file_dir, sep =',',index_col=0)
            self._df_factor_return.index = self._df_factor_return.index.map(lambda idx: datetime.strptime(idx,"%Y-%m-%d"))
            self._df_factor_return.sort_index(inplace=True)
        if config is not None:
            self.config = config
        if error_periods is not None:
            self.error_periods = error_periods
        if error_method is not None:
            self.error_method = error_method
        self._estimation = self.get_estimation()
        if self.error_method == "rolling_window":
            self._error = self.get_standard_error(periods=self.error_periods)
        elif self.error_method == "exponential":
            self._error = self.get_exponential_decay_error(periods=self.error_periods)


    def get_estimation(self):
        res = {}
        for method in self.config.keys():
            res[method] = self.config[method](self._df_factor_return)
        return res
    
    def add_view(self, config):
        for method in config.keys():
            self.config[method] = config[method]
            self._estimation[method] = config[method](self._df_factor_return)
        if self.error_method == "rolling_window":
            self._error = self.get_standard_error(periods=self.error_periods)
        elif self.error_method == "exponential":
            self._error = self.get_exponential_decay_error(periods=self.error_periods)


    def get_standard_error(self, periods=60):
        res = {}
        for method in self.config.keys():
            res[method] = pd.rolling_apply(
                            (self._df_factor_return - self._estimation[method]).shift(1).dropna(axis=0),
                            periods,
                            lambda x:np.sqrt(np.mean(x**2))
                                           ).dropna(axis=0)
        return res

    def get_exponential_decay_error(self, periods=60):
        res = {}
        for method in self.config.keys():
            res[method] = pd.ewma(((self._df_factor_return - self._estimation[method]).shift(1)**2).dropna
                           (axis=0), halflife=periods).dropna(axis=0)
            res[method] = np.sqrt(res[method])
        return res

    @property
    def methods(self):
        return self.config.keys()

    @property
    def factor_return(self):
        return deepcopy(self._df_factor_return)

    @property
    def estimation(self):
        return deepcopy(self._estimation)

    @property
    def error(self):
        return deepcopy(self._error)

    def __call__(self, method, target_date):
        EST_temp = self.estimation[method]
        ERR_temp = self.error[method]
        return deepcopy(EST_temp[EST_temp.index==target_date]), deepcopy(ERR_temp[ERR_temp.index==target_date])


class factor_return_data_reader():

    def __init__(self, file_dir, look_back_periods=60):
        self._df_factor_return = pd.read_csv(file_dir, sep =',',index_col=0)
        self._df_factor_return.index = self._df_factor_return.index.map(lambda idx: datetime.strptime(idx,"%Y-%m-%d"))
        self._df_factor_return.sort_index(inplace=True)
        self._look_back_periods = look_back_periods
        self.get_mean()

    def get_mean(self):
        self.df_mean = pd.rolling_mean(self._df_factor_return.shift(1), self._look_back_periods).dropna(axis=0)

    def __call__(self, target_date):
        start_date = target_date - timedelta(days=self._look_back_periods)
        df_temp = self._df_factor_return[(self._df_factor_return.index<target_date)&
                                         (self._df_factor_return.index>=start_date)]
        cov_mat = df_temp.cov()
        return deepcopy(self.df_mean[self.df_mean.index==target_date]), cov_mat


class factor_loading_data_reader():
    
    def __init__(self, file_dir):
        self._df_factor_loading = pd.read_csv(file_dir, sep =',',index_col=0)
        self._df_factor_loading.index = self._df_factor_loading.index.map(lambda idx: datetime.strptime(idx,"%Y-%m-%d"))
        self._df_factor_loading.sort_index(inplace=True)

    def __call__(self, target_date):
        df_temp = deepcopy(self._df_factor_loading[self._df_factor_loading.index==target_date]).dropna(axis=0).set_index('PERMNO')
        return df_temp.drop('RETNEXT', axis=1), df_temp[['RETNEXT']]

class residual_data_reader():
    
    def __init__(self, file_dir, look_back_periods=60, threshold = None, min_periods=2, clipped=True,
                                            diagonalized=True):
        self._df_residual = pd.read_csv(file_dir, sep =',',index_col=0)
        self._df_residual.index = self._df_residual.index.map(lambda idx: datetime.strptime(idx,"%Y-%m-%d"))
        self._df_residual.sort_index(inplace=True)
        self._look_back_periods = look_back_periods
        self._threshold = threshold
        self._min_periods = min_periods
        self._clipped = clipped
        self._diagonalized=diagonalized
    
    @property
    def residual(self):
        return deep_copy(self._df_residual)
    
    def set_threshold(self, threshold):
        self._threshold = threshold
    
    def set_min_periods(self, min_periods):
        self._min_periods = min_periods
    
    def set_look_back_period(self, look_back_periods):
        self._look_back_periods = look_back_periods
    
    def __call__(self, target_date, verbosity=False, clipped=True):
        end_date = target_date - timedelta(days=self._look_back_periods)
        df_temp = self._df_residual[(self._df_residual.index<target_date)&
                                    (self._df_residual.index>=end_date)]
        df_temp.reset_index(inplace=True)
        df_pivot = df_temp.pivot(index='index', columns='PERMNO', values='RESIDUAL')

        if self._diagonalized:
            start = timeit.default_timer()
            df_var = df_pivot.var()
            if self._threshold is not None:
                if self._clipped:
                    df_var = df_var.clip(self._threshold,1)
                else:
                    df_var = df_var[df_var>self]
            df_cov = pd.DataFrame(np.diag(df_var.as_matrix()), columns=df_var.index, index=df_var.index)
            end_cov = timeit.default_timer()
            if verbosity:
                print 'time for covariance calculation when diagonalized', end_cov - start
            return df_cov

        start = timeit.default_timer()
        df_cov = df_pivot.cov(min_periods=self._min_periods)
        end_cov = timeit.default_timer()
        if self._threshold is not None:
            if self._clipped:
                return df_cov.clip(self._threshold, 1)
            low_vol_stock_list = []
            for stock in df_cov.columns:
                if df_cov.at[stock, stock]<=self._threshold:
                    low_vol_stock_list.append(stock)
            trading_universe = list(set(df_cov.columns) - set(low_vol_stock_list))
            df_cov = df_cov.ix[trading_universe][trading_universe]
        end_threshold = timeit.default_timer()
        if verbosity:
            print 'time for covariance calculation', end_cov - start
            print 'time for threshold calculation', end_threshold - end_cov
        return df_cov

def examples():
    
    ###view data reader example
    def FACTOR_EMA(df, half_life):
        return pd.ewma(df, halflife=half_life)

    view_config = {"MA5" : lambda df: FACTOR_EMA(df.shift(1).dropna(axis=0), 5),
                   "MA30" : lambda df: FACTOR_EMA(df.shift(1).dropna(axis=0), 30),
                   "MA120" : lambda df: FACTOR_EMA(df.shift(1).dropna(axis=0), 120)}
    #
    file_dir = "Data/factor_return_w_industry.csv"
    test_view = view_data_reader(file_dir, view_config, error_method="exponential")
    # print 'Methods used:\n'
    # print test_view.methods
    EST, ERR = test_view('MA5',datetime(2014,1,3))
    print EST
    # print 'Estimations:\n', EST
    # print 'Error:\n', ERR

    ###factor return data reader example
    file_dir = "Data/factor_return_w_industry.csv"
    test_factor_return = factor_return_data_reader(file_dir)

    MEAN, COV = test_factor_return(datetime(2014,5,1))
    print 'Mean:\n', MEAN
    print 'Covariance:\n', COV
    print MEAN.shape
    print COV.shape

    # get factor loading and return of each stock
    DR_residual = residual_data_reader("Data/residual.csv",
                                       look_back_periods=60, threshold=None, min_periods=2, diagonalized=True)
    DR_residual.set_threshold(0.000001)
    RESIDUAL_COV = DR_residual(datetime(2014,5,1)).dropna(axis=0).dropna(axis=1)

    print RESIDUAL_COV.head(5)
    print RESIDUAL_COV.shape

    print DR_residual(datetime(2014, 5, 1)).dropna(axis=0).dropna(axis=1)

    ###factor loading data reader example
    file_dir = "Data/factor_loading_w_industry.csv"
    test_factor_loading = factor_loading_data_reader(file_dir)
    FACTOR_LOADING, STOCK_RETURN = test_factor_loading(datetime(2014,5,1))
    print 'Factor loading:\n', FACTOR_LOADING
    print 'Stock return:\n', STOCK_RETURN
    print FACTOR_LOADING.shape
    print STOCK_RETURN.shape

    TRADING_UNIVERSE = list(set.intersection(set(FACTOR_LOADING.index), set(RESIDUAL_COV.index)))
    D = RESIDUAL_COV.ix[TRADING_UNIVERSE][TRADING_UNIVERSE].as_matrix()

    if 'NAME' in FACTOR_LOADING.columns:
        FACTOR_LOADING.drop('NAME', axis=1, inplace=True)

    # FACTOR_ALL = list(MEAN.columns)
    FACTOR_ALL = list(MEAN.columns)

    # factor loading matrix
    X = FACTOR_LOADING.ix[TRADING_UNIVERSE][FACTOR_ALL].as_matrix()

    # prior mean of the factors
    XI = MEAN[FACTOR_ALL].as_matrix().transpose()

    # prior convariance matrix
    V = COV.ix[FACTOR_ALL][FACTOR_ALL].as_matrix()

    print "last piece of sanity check"
    print FACTOR_LOADING.shape
    print COV.shape
    print RESIDUAL_COV.shape
    part1 = FACTOR_LOADING.dot(COV).dot(FACTOR_LOADING.transpose())
    part2 = RESIDUAL_COV

    print part1.shape
    print part2.shape

    part3 = X.dot(V).dot(X.transpose()) + D
    print part3.shape

    part3_inv = np.linalg.inv(part3)
    print part3_inv.shape

    stock_t = STOCK_RETURN.ix[TRADING_UNIVERSE].as_matrix().reshape(len(TRADING_UNIVERSE))

    part4 = np.matmul(part3_inv, X)
    h_markwotiz = np.matmul(part4, XI)

    holding_markwotiz = np.matmul(np.matmul(part3_inv, X), XI)
    ret_markwotiz = h_markwotiz.reshape(len(TRADING_UNIVERSE)).dot(stock_t)
    return_markwotiz = holding_markwotiz.reshape(len(TRADING_UNIVERSE)).dot(stock_t)


    VAR = part1 + part2
    print VAR.shape


    ###residual data reader example
    # df_sample_residual = pd.DataFrame({'Date': [datetime(2015,5,5),datetime(2015,5,5), datetime(2015,5,5),
    #                                           datetime(2015,5,6),datetime(2015,5,6),datetime(2015,5,6)],
    #               'PERMNO': ['10107', '90319', '12060', '10107', '90319', '12060'],
    #               'RESIDUAL': [0.1, -0.2, 0.1, 0.4, -0.5, 0.6]}).set_index('Date')
    #
    # df_sample_residual.to_csv("sample_residual.csv")
    # test_residual = residual_data_reader("sample_residual.csv")
    # print 'Residual covariance:\n', test_residual(datetime(2015,5,7))
    # test_residual.set_threshold(0.05)
    # print 'After remove extreme low vol stocks:\n', test_residual(datetime(2015,5,7))


if __name__ == "__main__":
    examples()
####Some preprocessing
#    df_factor_loading=pd.read_csv("/Users/wyx/Documents/capstone/Data/factor_loading_raw.csv", sep =',')
#    df_factor_loading = df_factor_loading.drop(['Unnamed: 0', 'XDATE','RDATE'], axis=1).rename(columns={'T':'Date'}).set_index('Date')
#    df_factor_loading = df_factor_loading[df_factor_loading['INDUSTRY']!='Unknown']
#    
#    for industry in set(df_factor_loading['INDUSTRY']):
#        df_factor_loading[industry] = (df_factor_loading['INDUSTRY']==industry).astype(int)
#        #df_factor_loading.apply(lambda row: 1 if row['INDUSTRY']==industry else 0, axis=1)
#    
#    df_factor_loading.drop('INDUSTRY', axis=1).to_csv("/Users/wyx/Documents/capstone/Data/factor_loading_w_industry.csv")
#
#     df_residual = pd.read_csv("/Users/wyx/Documents/capstone/Data/residual_return_v2.csv", sep =',')
#     df_residual = df_residual.drop(['Unnamed: 0'],axis=1).rename(columns={'Ticker':'PERMNO',
#                                                             'Residual':'RESIDUAL'}).set_index('Date')





