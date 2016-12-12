from r_py_ts import *

class var(ts_generic):
    """
        incomplete !!!
    """
    def __init__(self, df, lag=1, pred_step = 1):
        self.data = df
        self.step = pred_step
        self.lag = lag
        self.result = None

    def var_predict_next(self, sub_df):
        """
        :param sub_df: here sub_df could be multiple column dataframe
        :return:
        """
        base.rm('tmp')
        print sub_df
        self.py_to_ts_zoo(sub_df, 'tmp')
        print type(ro.globalenv['tmp'])
        print ro.globalenv['tmp']
        func_param = "as.data.frame(fortify(predict(VAR(tmp,p=lag,type=\"const\"), n.ahead=step)))"\
            .replace("lag", str(self.lag))\
            .replace("step", str(self.step))

        print func_param
        pred_step = ro.r(func_param)
        print type(pred_step)
        print self.r_to_py(pred_step)
        return self.r_to_py(pred_step)
        # base.rm('\'' + str(pred_step) + '\'')  # clean the memory


    def __call__(self, *args, **kwargs):
        return self.hw_predict_df()