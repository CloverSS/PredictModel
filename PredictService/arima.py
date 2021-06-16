import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller as ADF
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.arima_model import ARMA
from pmdarima import auto_arima


class Arima:

    def __init__(self, history_data):
        self.history_data = history_data.data

    def auto_arima(self):
        future_len = 10
        data = pd.DataFrame(columns=['timestamp', 'value'])
        for history_struct in self.history_data:
             data = data.append({'timestamp':(int)(history_struct.timestamp), 'value':history_struct.value}, ignore_index=True) 
        train = data['value']
        model = auto_arima(train, trace=True, error_action='ignore', suppress_warnings=True)
        model.fit(train)
        
        forecast = model.predict(n_periods=future_len)
        print(forecast)
        return forecast
        # self.plot_results(forecast, valid)
        # forecast = pd.DataFrame(forecast, index = valid.index, columns=['Prediction'])

    def arma_predict(self, future_len):
        data = list(self.raw_data[self.value_name])
        """
        import statsmodels.tsa.stattools as st
        order = st.arma_order_select_ic(data,max_ar=2,max_ma=2,ic=['aic', 'bic', 'hqic'])
        """
        model = ARMA(data[:-future_len], order=(0, 2))
        result_arma = model.fit(disp=-1, method='css')
        predict = result_arma.predict(len(data)-future_len, len(data))
        RMSE = np.sqrt(((predict-data[len(data)-future_len-1:])**2).sum()/(future_len+1))
        self.plot_results(predict, data[len(data)-future_len-1:])
        print(RMSE)
        return predict, RMSE
