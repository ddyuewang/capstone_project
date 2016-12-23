# Wrapper on R dataframe Integration with python

# all necessary Python library

import numpy as np
import pandas as pd
import rpy2
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
import rpy2.robjects as ro
import pandas.rpy.common as com
import warnings
import copy
warnings.filterwarnings("ignore")
warnings.simplefilter(action = "ignore", category = FutureWarning)

# all necessary R libraries

stats = importr('stats')
base = importr('base')
datasets = importr('datasets')
graph = importr('graphics')
forecast = importr('forecast')
fortify = importr('ggfortify')
vars = importr('vars')
zoo = importr('zoo')

ro.r('library(forecast)')

class ts_generic():
    def __init__(self):
        pass

    @staticmethod
    def py_to_r(df, r_df_name):
        """
        :param df:  pandas data frame
        :param name:  saved r data frame in R environment
        :return:
        """
        r_df = pandas2ri.py2ri(df)
        ro.globalenv[r_df_name] = r_df
        return None

    @staticmethod
    def r_to_py(r_df_name):
        """
        :param r_df_name: the r data frame NAME saved in R environment
        :return:
        """
        ro.globalenv['\'' + str(r_df_name) + '\''] = r_df_name
        return com.load_data('\'' + str(r_df_name) + '\'')

    @staticmethod
    def r_to_ts(r_df_name, r_ts_name, freq=365):
        """
        :param r_df_name:  convert python data frame to r ts
        :return:
        """
        r_ts = stats.ts(r_df_name, start=1, frequency=freq)
        ro.globalenv[r_ts_name] = r_ts
        return None

    @staticmethod
    def py_to_ts(df, r_ts_name, freq=365):
        """
        :param r_df_name:  convert a R data frame to r ts - dirct
        :return:
        """
        r_df_name = pandas2ri.py2ri(df)
        r_ts = stats.ts(r_df_name, start=1, frequency=freq)
        ro.globalenv[r_ts_name] = r_ts
        return None

    @staticmethod
    def py_to_ts_zoo(df, r_ts_name, freq=365):
        """
        default to be a daily index  - o/w needs to change the format parameters
        :param r_df_name:  convert a R data frame to r ts
        :return:

        """
        r_df_name = pandas2ri.py2ri(df)
        ro.globalenv['tmp_ts'] = r_df_name
        r_ts = ro.r('zoo(tmp_ts)')
        ro.globalenv[r_ts_name] = stats.ts(r_ts)
        return None

class arima(ts_generic):

    def __init__(self, df, lookback=200, pred_step=1):
        self.data = df
        self.step = pred_step
        self.insample = lookback
        self.result = None

    def arima_predict_next(self, sub_df):
        """
        :param sub_df: here sub_df is one column dataframe
        :return:
        """
        base.rm('tmp')
        self.py_to_r(sub_df, 'tmp')
        func_param = "as.data.frame(forecast(auto.arima(tmp),h=step))".replace("step", str(self.step))
        pred_step = ro.r(func_param)
        return self.r_to_py(pred_step)


    def arima_predict_df(self):
        """
        # method to wrap the udpdate on the dataframe level
        :param master_df:
        :param in_sample_count:
        :return:
        """
        df_init = pd.DataFrame(index=self.data.index, columns=self.data.columns)
        df_init.iloc[:self.insample, :] = self.data.iloc[:self.insample, :]

        # for col in df_init.columns:
        #     for row in df_init.iloc[self.insample:,:].index:
        #
        #         #### old version
        #         # tmp_df = df_init[[col]].dropna()
        #         # tmp_df[[col]] = tmp_df[[col]].astype(float)
        #         # tmp_res = self.arima_predict_next(tmp_df[col])
        #         # df_init.loc[row, [col]] = tmp_res.iloc[0]['Point Forecast']
        #         # del tmp_df
        #
        #         # # update each entries by entries by calling single predict function
        #         # tmp_df = df_init[[col]].dropna()

        col_name = 0
        for col in df_init.columns:
            col_name = col_name + 1
            for row in range(self.insample, len(df_init.index)+1):
                start = row - self.insample
                tmp_df = copy.deepcopy(self.data.iloc[start:row,:])
                tmp_df = tmp_df[[col]]
                tmp_df[[col]] = tmp_df[[col]].astype(float)
                print (col_name, tmp_df.index[-1])
                tmp_res = self.arima_predict_next(tmp_df[col])
                df_init.loc[tmp_df.index[-1],[col]] = tmp_res.iloc[0]['Point Forecast']
                del tmp_df
                del tmp_res

        self.result = df_init

    def __call__(self, *args, **kwargs):
        return self.arima_predict_df()


class holt_winter(ts_generic):

    def __init__(self, df, lookback=200, pred_step=1):
        self.data = df
        self.step = pred_step
        self.insample = lookback
        self.result = None

    def hw_predict_next(self, sub_df):
        """
        :param sub_df: here sub_df is one column dataframe
        :return:
        """
        base.rm('tmp')
        self.py_to_ts(sub_df, 'tmp')
        func_param = "as.data.frame(forecast(HoltWinters(tmp,gamma = FALSE),h=step))".replace("step", str(self.step))
        pred_step = ro.r(func_param)
        return self.r_to_py(pred_step)

    def hw_predict_df(self):
        """
        # method to wrap the udpdate on the dataframe level
        :param master_df:
        :param in_sample_count:
        :return:
        """
        df_init = pd.DataFrame(index=self.data.index, columns=self.data.columns)
        df_init.iloc[:self.insample, :] = self.data.iloc[:self.insample, :]

        # for col in df_init.columns:
        #     for row in df_init.iloc[self.insample:, :].index:
        #         # update each entries by entries by calling single predict function
        #         # print (row, col)
        #         tmp_df = df_init[[col]].dropna()
        #         tmp_df[[col]] = tmp_df[[col]].astype(float)
        #         tmp_res = self.hw_predict_next(tmp_df[col])
        #         df_init.ix[tmp_df.index[-1], [col]] = tmp_res.iloc[0]['Point Forecast']
        #         del tmp_df
        col_name = 0
        for col in df_init.columns:
            col_name = col_name + 1
            for row in range(self.insample, len(df_init.index) + 1):
                start = row - self.insample
                tmp_df = copy.deepcopy(self.data.iloc[start:row, :])
                tmp_df = tmp_df[[col]]
                tmp_df[[col]] = tmp_df[[col]].astype(float)
                tmp_res = self.hw_predict_next(tmp_df[col])
                print (col_name, tmp_df.index[-1])
                df_init.loc[tmp_df.index[-1], [col]] = tmp_res.iloc[0]['Point Forecast']
                del tmp_df
                del tmp_res

        self.result = df_init

    def __call__(self, *args, **kwargs):
        return self.hw_predict_df()

if __name__ == "__main__":

    ## test arima data frame version ###
    np.random.seed(0)
    count = np.random.rand(457,2)
    count = count/100
    df = pd.DataFrame(index=pd.date_range('2016-01-01', '2017-04-01'), data=count, columns=['count1','count2'])
    print df.shape
    print df.head(5)
    arima_test = arima(df, lookback=420)
    arima_test()
    print arima_test.result

    ### test arima data frame version ###
    # np.random.seed(0)
    # count = np.random.rand(457,2)
    # df = pd.DataFrame(index=pd.date_range('2016-01-01', '2017-04-01'), data=count, columns=['count1','count2'])
    # print df.shape
    # print df.tail(5)
    # hw_test = holt_winter(df, lookback=450)
    # hw_test()
    # print hw_test.result


