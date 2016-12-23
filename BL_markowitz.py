#BL model class
from BL_data_reader import view_data_reader, factor_return_data_reader,\
                        factor_loading_data_reader, residual_data_reader
import py_ts as ts_py
import r_py_ts as ts_r
from datetime import datetime, timedelta
import BL_data_reader as BLDR
import numpy as np
import pandas as pd
import timeit

class Markowitze_Portfolio():

    def __init__(self, data_reader_config, risk_aversion):
        # self.view = data_reader_config['view']
        # self._factor_diagonalized = factor_diagonalized
        self.factor_return = data_reader_config['factor_return']
        self.residual = data_reader_config['residual']
        self.factor_loading = data_reader_config['factor_loading']
        self._risk_aversion = risk_aversion

    def set_risk_aversion(self, risk_aversion):
        self._risk_aversion = risk_aversion

    def single_period(self, target_date, verbosity=False):

        #get factor posterior
        MEAN, COV = self.factor_return(target_date)

        #get factor loading and return of each stock
        FACTOR_LOADING, STOCK_RETURN = self.factor_loading(target_date)

        #get residul covariance
        RESIDUAL_COV = self.residual(target_date).dropna(axis=0).dropna(axis=1)
        is_valid_date = min([MEAN.shape[0], COV.shape[0],
                             FACTOR_LOADING.shape[0], STOCK_RETURN.shape[0],
                             RESIDUAL_COV.shape[0]])>0

        if not is_valid_date:
            if verbosity:
                print target_date ,'is not a trading day'
            return None, None, None

        ### the implementation of markowitz portfolio

        # NUM_PREDICTOR = len(self.view.methods)
        # NUM_FACTOR = EST.shape[1]
        # M = np.kron(np.array([[1]] * NUM_PREDICTOR),np.eye(NUM_FACTOR))

        TRADING_UNIVERSE = list(set.intersection(set(FACTOR_LOADING.index),set(RESIDUAL_COV.index)))
        D = RESIDUAL_COV.ix[TRADING_UNIVERSE][TRADING_UNIVERSE].as_matrix()

        if 'NAME' in FACTOR_LOADING.columns:
            FACTOR_LOADING.drop('NAME', axis=1, inplace=True)

        # FACTOR_ALL = list(MEAN.columns)
        FACTOR_ALL = list(MEAN.columns)

        #factor loading matrix
        X = FACTOR_LOADING.ix[TRADING_UNIVERSE][FACTOR_ALL].as_matrix()

        #prior mean of the factors
        XI = MEAN[FACTOR_ALL].as_matrix().transpose()

        #prior covariance of factors
        V = COV.ix[FACTOR_ALL][FACTOR_ALL].as_matrix()

        #construct the markowitz portfolio

        omega =  X.dot(V).dot(X.transpose()) + D
        omega_inv = np.linalg.inv(omega)
        stock_t = STOCK_RETURN.ix[TRADING_UNIVERSE].as_matrix().reshape(len(TRADING_UNIVERSE))

        h_markwotiz = np.matmul(np.matmul(omega_inv, X), XI)
        ret_markwotiz = h_markwotiz.reshape(len(TRADING_UNIVERSE)).dot(stock_t)

        return ret_markwotiz, h_markwotiz, TRADING_UNIVERSE

        # # todo: make the matrix calculation more efficient!
        # SIGMA = D + np.matmul(np.matmul(X, V),X.transpose())
        # SIGMA_INV = np.linalg.inv(SIGMA)
        # OMEGA_INV = np.linalg.inv(OMEGA)
        # M_OMEGA_INV = np.matmul(M.transpose(), OMEGA_INV)#for convenience
        # V_INV = np.linalg.inv(V)
        # X_SIGMA_INV = np.matmul(X.transpose(), SIGMA_INV)#for convenience
        #
        # part1 = V_INV + np.matmul(M_OMEGA_INV, M) + np.matmul(X_SIGMA_INV, X)
        # part2 = np.matmul(V_INV, XI) + np.matmul(M_OMEGA_INV, Q)
        # h_blb = 1.0 / self._risk_aversion * np.matmul(X_SIGMA_INV.transpose(),
        #                                              np.matmul(np.linalg.inv(part1), part2))
        # ret_blb = h_blb.reshape(len(TRADING_UNIVERSE)).dot(STOCK_RETURN.ix[TRADING_UNIVERSE].as_matrix().reshape(len(TRADING_UNIVERSE)))
        #
        # return ret_blb, h_blb, TRADING_UNIVERSE

    def __call__(self, start_date, end_date):
        day = start_date
        pnl = [0]
        date_list = [start_date]
        while day <= end_date:
            day += timedelta(days=1)
            ret, _, _ = self.single_period(day)
            if ret is not None:
                pnl.append(pnl[-1]+ret)
                date_list.append(day)
        return pd.DataFrame({'Date':date_list,
                             'pnl': pnl}).set_index('Date')


def main():


    #you can add new view
    # def dummy_predictor(csv_file):
    #     #using saved calculation result
    #     _df_factor_return = pd.read_csv(csv_file, sep =',',index_col=0)
    #     _df_factor_return.index = _df_factor_return.index.map(lambda idx: datetime.strptime(idx,"%Y-%m-%d"))
    #     return _df_factor_return
    #
    # new_view = {"GRU": lambda df: dummy_predictor("Prediction/prediction_rnn_gru.csv")}
    #DR_view.add_view(new_view)

    DR_factor_return = BLDR.factor_return_data_reader("Data/factor_return_w_industry.csv",
                                                  look_back_periods=60)
    DR_residual = BLDR.residual_data_reader("Data/residual.csv",
                                        look_back_periods=60, threshold = None, min_periods=2, diagonalized=True)
    DR_residual.set_threshold(0.000001)
    DR_factor_loading = BLDR.factor_loading_data_reader("Data/factor_loading_w_industry.csv")

    data_reader_config = {#'view': DR_view, #call method return estimation and standard error of each predictors
        'factor_return': DR_factor_return, #call method return prior mean and covariance of factors
            'residual': DR_residual,#call method return covariance of residuals
                'factor_loading': DR_factor_loading #call method return factor loadings of each stock, also return the return of next B day for each stock
                }

    # BLP = Black_Litterman_Portfolio(data_reader_config, 1, factor_diagonalized=False)
    # print BLP.view.methods

    markowitz_p = Markowitze_Portfolio(data_reader_config, 1)

    target_date = datetime(2015,5,8)
    MEAN, COV = markowitz_p.factor_return(target_date)
    FACTOR_LOADING, STOCK_RETURN = markowitz_p.factor_loading(target_date)
    RESIDUAL_COV = markowitz_p.residual(target_date)

    ret_markowitz, h_markowitz, TRADING_UNIVERSE = markowitz_p.single_period(target_date)
    print 'Dummy pnl', markowitz_p(datetime(2006,5,1), datetime(2006,5,12))


if __name__ == "__main__":
    main()

