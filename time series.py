# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 15:36:17 2019

@author: rxu
"""

import pandas as pd
import warnings
import itertools
import numpy as np
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
plt.style.use('fivethirtyeight')
import pandas as pd
import statsmodels.api as sm
import matplotlib
import os.path

fullpath = os.path.abspath('..\\data\\time series.xlsx')

df = pd.read_excel(fullpath, index = False)

df = df[df['StoreCountryCode'] == 'US']

df_dyn = df[df['BannerName'] == 'Dy'][['New_Cust_Count', 'FirstTransactionDatetime']].reset_index()

df_grg = df[df['BannerName'] == 'Ga'][['New_Cust_Count', 'FirstTransactionDatetime']].reset_index()

##############

df_dyn = df_dyn.set_index('FirstTransactionDatetime')
df_dyn.index

y = df_dyn['New_Cust_Count'].resample('W').sum().fillna(df_dyn['New_Cust_Count'].mean())

from pylab import rcParams
rcParams['figure.figsize'] = 18, 8
decomposition = sm.tsa.seasonal_decompose(y, model='additive')
fig = decomposition.plot()
plt.show()


#p = d = q = range(0, 3)
#pdq = list(itertools.product(p, d, q))
#seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
#
#param_list = []
#param_seasonal_list = []
#results = []
#
#for param in pdq:
#    for param_seasonal in seasonal_pdq:
#        try:
#            mod = sm.tsa.statespace.SARIMAX(y,
#                                            order=param,
#                                            seasonal_order=param_seasonal,
#                                            enforce_stationarity=False,
#                                            enforce_invertibility=False)
#            
#            results = mod.fit()
#            print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
#        except:
#            continue



        
#####
mod = sm.tsa.statespace.SARIMAX(y,
                                order=(0,1,1),
                                seasonal_order=(2, 2, 1, 12),
                                enforce_stationarity=False,
                                enforce_invertibility=False)
results = mod.fit()
print(results.summary().tables[1])


###
pred = results.get_prediction(start=pd.to_datetime('2019-07-07'), dynamic=False)
pred_ci = pred.conf_int()
ax = y['2018':].plot(label='observed')
pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7, figsize=(14, 7))
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.2)
ax.set_xlabel('Date')
ax.set_ylabel('Furniture Sales')
plt.legend()
plt.show()



pred_uc = results.get_forecast(steps=15)
pred_ci = pred_uc.conf_int()
ax = y.plot(label='observed', figsize=(14, 7))
pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.25)
ax.set_xlabel('Date')
ax.set_ylabel('Furniture Sales')
plt.legend()
plt.show()