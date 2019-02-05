# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 16:07:35 2019

@author: SarachErudite
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import argrelextrema

df = pd.read_csv("Dataset/SEHK_30Min_2800.csv")

n=11*60 # number of points to be checked before and after

df['Min'] = df.iloc[argrelextrema(df.Close.values, np.less_equal, order=n)[0]]['Close']
df['Max'] = df.iloc[argrelextrema(df.Close.values, np.greater_equal, order=n)[0]]['Close']

plt.scatter(df.index, df['Min'], c='r')
plt.scatter(df.index, df['Max'], c='g')
plt.plot(df.index, df['Close'])
plt.show()

#df.to_csv("Dataset/SEHK_30Min_2800.csv",index=False)