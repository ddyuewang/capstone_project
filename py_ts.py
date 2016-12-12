import pandas as pd
import numpy as np
import copy
from statsmodels.tsa.vector_ar.dynamic import DynamicVAR

import warnings
warnings.filterwarnings("ignore")


class vector_ar():

    def __init__(self, df, lookback_step=200 ,pred_step=1, ):
        self.data = df
        self.step = pred_step
        self.in_sample = lookback_step
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
        df_init = copy.deepcopy(self.data)
        for row in range(self.in_sample+1, len(df_init.index)):

            tmp_df = df_init.iloc[:row, :]
            df_init.iloc[row,:] = self.var_predict_next(tmp_df).iloc[0,:]

        self.result = df_init
        return None

    def __call__(self, *args, **kwargs):
        self.var_predict_df()

if __name__ == "__main__":

    count = np.random.rand(214, 3)
    df = pd.DataFrame(index=pd.date_range('2016-01-01', '2016-08-01'), data=count, columns=['count1','count2','count3'])
    var_test = vector_ar(df)
    var_test()
    print var_test.result.iloc[-6:,:]
