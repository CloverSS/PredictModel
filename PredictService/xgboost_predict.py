import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
from time import time
import numpy

class Xgboost:

    def __init__(self, history_data):
        self.history_data = history_data.data

    def create_features(self, df, label=None):
        df['timestamp'] = df.index
        df['hour'] = df['timestamp'].dt.hour
        df['min'] = df['timestamp'].dt.minute
        df['dayofyear'] = df['timestamp'].dt.dayofyear
        
        X = df[['hour', 'min', 'dayofyear']]
        if label:
            y = df[label]
            return X, y
        return X

    def predict(self):
        data = pd.DataFrame(columns=['timestamp', 'value'])
        for history_struct in self.history_data:
             data = data.append({'timestamp':(int)(history_struct.timestamp), 'value':history_struct.value}, ignore_index=True)
        predict_start = (int)(self.history_data[-1].timestamp)
        predict_data = pd.DataFrame()
        predict_data['timestamp'] = numpy.arange(predict_start+30, predict_start+60*10+30, step = 30, dtype = None)
       
        predict_data['timestamp'] = pd.to_datetime(predict_data['timestamp'], unit='s')
        predict_data = predict_data.set_index("timestamp")
       
        data['timestamp'] = pd.to_datetime(data['timestamp'], unit='s')
        data = data.set_index("timestamp")
        
        '''
        _ = data_test \
            .rename(columns={'y': 'TEST SET'}) \
            .join(data_train.rename(columns={'y': 'TRAINING SET'}), how='outer') \
            .plot(figsize=(15, 5), title='PJM East', style='.')
        '''

        X_train, y_train = self.create_features(data, label='value')
        predict_data = self.create_features(predict_data)
       
        xg_model = xgb.XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1, colsample_bytree=1, gamma=0, learning_rate=0.1, max_delta_step=0, max_depth=3, min_child_weight=1, missing=None, n_estimators=1000, n_jobs=1, nthread=None, objective='reg:linear', random_state=0, reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None, silent=True, subsample=1)
        xg_model.fit(X_train, y_train, eval_set=[(X_train, y_train)], early_stopping_rounds=50, verbose=False) 

        predict_data['MW_Prediction'] = xg_model.predict(predict_data)
        print(predict_data['MW_Prediction'])
        return predict_data['MW_Prediction']