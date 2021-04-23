# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 19:51:47 2021

@author: Ruhul.Akhtar
"""

#Stock Price Analysis
 
import os
os.chdir("D:/KNOWLEDGE KORNER/ANALYTICS/MISC/Stock_Market")
get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import mplfinance as mpf
 
## Read file
file = "RELIANCE.NS.csv"
data = pd.read_csv(file)
data.info()
data.columns
data.Date = pd.to_datetime(data.Date)
data = data.set_index('Date')
data
 
mpf.plot(data)
mpf.plot(data, type='line', volume=True)
mpf.plot(data['2020-10'], volume=True)
mpf.plot(data['2020-10'], type='candle', volume=True)
mpf.plot(data['2020-08':'2020-11'], type='candle', mav=20, volume=True)
mpf.plot(data['2020-08':'2020-11'], type='candle', mav=20, volume=True, style='yahoo')
mpf.plot(data['2019-11':'2020-11'], type='candle', mav=(20,50,100), volume=True, style='yahoo')

