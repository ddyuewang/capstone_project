import pandas as pd
import numpy as np
import copy
from statsmodels.tsa.vector_ar.dynamic import DynamicVAR

import warnings
warnings.filterwarnings("ignore")


class vector_ar():
    def __init__(self, df, lookback=200, pred_step=1):
        self.data = df
        self.step = pred_step
        self.insample = lookback
        self.result = None

    def var_predict_next(self, sub_df):

        """
        :param sub_df: here sub_df is multi-dataframe with one extra column you want to predict
        :return:

        """
        var = DynamicVAR(sub_df, min_periods=sub_df.shape[0]-2, window_type='expanding')
        return var.forecast(self.step)

    def var_predict_df(self):
        """
        # method to wrap the udpdate on the dataframe level
        :param master_df:
        :param in_sample_count:
        :return:
        """
        # df_init = copy.deepcopy(self.data)
        df_init = pd.DataFrame(index=self.data.index, columns=self.data.columns)
        df_init.iloc[:self.insample, :] = self.data.iloc[:self.insample, :]

        for row in range(self.insample, len(df_init.index)+1):

            ## fix look back range
            start = row-self.insample
            tmp_df = copy.deepcopy(self.data.iloc[start:row, :])
            df_init.ix[tmp_df.index[-1], :] = self.var_predict_next(tmp_df).iloc[0,:]
            print tmp_df.index[-1]

        self.result = df_init
        del df_init
        return None

    def __call__(self, *args, **kwargs):
        self.var_predict_df()

if __name__ == "__main__":

    count = np.random.rand(214, 3)
    df = pd.DataFrame(index=pd.date_range('2016-01-01', '2016-08-01'), data=count, columns=['count1','count2','count3'])
    var_test = vector_ar(df,lookback=201)
    var_test()
    print df.iloc[-14:,:]
    print var_test.result.iloc[-14:,:]
