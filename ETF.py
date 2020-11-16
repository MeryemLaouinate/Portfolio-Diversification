from pandas_datareader import data as web
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
import streamlit as st

from pypfopt import risk_models
from pypfopt import expected_returns
from pypfopt import EfficientFrontier
from pypfopt import HRPOpt
from pypfopt import CLA
#from WebScrap import dftest

df_morocco = pd.read_excel('Bourse_De_Casa.xlsx')

#Get the stock symbols in the portfolio
#FAANG
assets = ['FB', 'AMZN','AAPL','NFLX','GOOG','IBM','MSFT', 'MCD','MA','TSLA','KO','NVDA','HD','ORCL','SAP',
          'CSCO','VMW','INTC','NOW','BLK','EBAY','DXC','VMC','VLO','WM','WST']


st.title('Portfolio Diversification')

st.write("""
# Explore different optimizer and datasets
""")

ETF_name = st.sidebar.multiselect('Select ETF stocks :', assets)

moroccan_stocks = st.sidebar.multiselect('Select moroccan stocks :', df_morocco.columns.difference(['Session']))

df_moroccan_stocks = df_morocco.loc[:,df_morocco.columns.isin(moroccan_stocks)]

#st.write(f"## {ETF_name} ETF")
model_name = st.sidebar.selectbox(
    'Select model',
    ('','CLA', 'EF', 'HRP')
)
#st.write(f"## {model_name} Model")


weights = np.random.random(len(ETF_name))
weights = weights/np.sum(weights)

wei = np.random.random(len(moroccan_stocks))
wei = wei/np.sum(wei)

#Get the stocks/portfolio starting date
stockStartDate = '2018-01-01'

#Get the stocks ending date (today)
today = datetime.today().strftime('%Y-%m-%d')

#Create a dataframe to store adjusted close price of the stocks
df_ETF = pd.DataFrame()

#Store the adjusted close price into the df
for stock in ETF_name :
  df_ETF[stock] = web.DataReader(stock, data_source='yahoo', start = stockStartDate, end = today)['Adj Close']

# Show the daily simple returns
returns = df_ETF.pct_change()
returns.dropna(inplace=True)

moroccan_stocks_returns = df_moroccan_stocks.pct_change()
moroccan_stocks_returns.dropna(inplace=True)

############################################# Efficient Frontier

def deviation_risk_parity(w, cov_matrix):
    diff = w * np.dot(cov_matrix, w) - (w * np.dot(cov_matrix, w)).reshape(-1, 1)
    return (diff ** 2).sum().sum()

def calculEF(dataframe):
    mu = expected_returns.mean_historical_return(dataframe)
    S = risk_models.sample_cov(dataframe)
    ef = EfficientFrontier(mu, S)
    weights = ef.nonconvex_objective(deviation_risk_parity, ef.cov_matrix)
    ef.portfolio_performance(verbose=True)
    return pd.DataFrame([weights])


colors_list = ['#5cb85c','#F9429E','#2C75FF','#DF73FF','#25FDE9','#660099']

if model_name == 'EF' :
    if len(ETF_name) == 0 & len(moroccan_stocks) == 0:
        st.write("Please select stocks!!!")
    if len(ETF_name) != 0 :
        df_ETF = calculEF(df_ETF)
        result_pct = df_ETF.div(df_ETF.sum(1), axis=0)
        ax = result_pct.plot(kind='bar', figsize=(20, 6), width=0.6, color=colors_list, edgecolor=None)
        plt.legend(labels=df_ETF.columns, fontsize=20)
        plt.xticks(fontsize=20)
        for spine in plt.gca().spines.values():
            spine.set_visible(False)
        plt.yticks([])
        # Add this loop to add the annotations
        for p in ax.patches:
            width = p.get_width()
            height = p.get_height()
            x, y = p.get_xy()
            ax.annotate(f'{height:.0%}', (x + width / 2, y + height), ha='center', fontsize=20)
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.write(f"## ETF Assets Allocation using Efficient Frontier")
        st.pyplot()
        st.write(f"## Max Sharp Ratio :  ")

    if len(moroccan_stocks) != 0:
        df_moroccan_stocks = calculEF(df_moroccan_stocks)
        result_pct = df_moroccan_stocks.div(df_moroccan_stocks.sum(1), axis=0)
        ax = result_pct.plot(kind='bar', figsize=(20, 6), width=0.6, color=colors_list, edgecolor=None)
        plt.legend(labels=df_moroccan_stocks.columns, fontsize=20)
        plt.xticks(fontsize=20)
        for spine in plt.gca().spines.values():
            spine.set_visible(False)
        plt.yticks([])
        # Add this loop to add the annotations
        for p in ax.patches:
            width = p.get_width()
            height = p.get_height()
            x, y = p.get_xy()
            ax.annotate(f'{height:.0%}', (x + width / 2, y + height), ha='center', fontsize=20)
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.write(f"## Moroccan Assets Allocation using Efficient Frontier")
        st.pyplot()
        st.write(f"## Max Sharp Ratio : ")
################################## Herarchical Risk Parity


if model_name == 'HRP':

    hrp_ETF = HRPOpt(returns)
    hrp_Mor = HRPOpt(moroccan_stocks_returns)

    weights = hrp_ETF.optimize(linkage_method='single')
    wei = hrp_Mor.optimize(linkage_method='single')

    hrp_ETF.portfolio_performance(verbose=True)
    hrp_Mor.portfolio_performance(verbose=True)

    dfhrpETF = pd.DataFrame([weights])
    dfhrpMor = pd.DataFrame([wei])

    if len(ETF_name) == 0 & len(moroccan_stocks) == 0:
        st.write("Please select stocks!!!")
    if len(ETF_name) != 0:
        result_pct = dfhrpETF.div(dfhrpETF.sum(1), axis=0)
        ax = result_pct.plot(kind='bar', figsize=(20, 6), width=0.8, color=colors_list, edgecolor=None)
        plt.legend(labels=dfhrpETF.columns, fontsize=20)
        plt.xticks(fontsize=20)
        for spine in plt.gca().spines.values():
            spine.set_visible(False)
        plt.yticks([])
        # Add this loop to add the annotations
        for p in ax.patches:
            width = p.get_width()
            height = p.get_height()
            x, y = p.get_xy()
            ax.annotate(f'{height:.0%}', (x + width / 2, y + height * 1.02), ha='center', fontsize=20)
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.write(f"## ETF Assets Allocation using Hierarchical Risk Parity")
    if len(moroccan_stocks) != 0:
        result_pct = dfhrpMor.div(dfhrpMor.sum(1), axis=0)
        ax = result_pct.plot(kind='bar', figsize=(20, 6), width=0.8, color=colors_list, edgecolor=None)
        plt.legend(labels=dfhrpMor.columns, fontsize=20)
        plt.xticks(fontsize=20)
        for spine in plt.gca().spines.values():
            spine.set_visible(False)
        plt.yticks([])
        # Add this loop to add the annotations
        for p in ax.patches:
            width = p.get_width()
            height = p.get_height()
            x, y = p.get_xy()
            ax.annotate(f'{height:.0%}', (x + width / 2, y + height * 1.02), ha='center', fontsize=20)
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.write(f"## Moroccan Assets Allocation using Hierarchical Risk Parity")
        st.pyplot()

############################################ CRITICAL LINE ALGORITHM

def calculCLA(dataframe):
    mu = expected_returns.mean_historical_return(dataframe)
    S = risk_models.sample_cov(dataframe)
    cla = CLA(mu, S)
    weights = cla.max_sharpe()

    return pd.DataFrame([weights])


if model_name == 'CLA':
    if len(ETF_name) == 0 & len(moroccan_stocks) == 0:
        st.write("Please select stocks!!!")
    if len(ETF_name) != 0 :
        df_ETF = calculCLA(df_ETF)
        result_pct = df_ETF.div(df_ETF.sum(1), axis=0)
        ax = result_pct.plot(kind='bar', figsize=(20, 6), width=0.6, color=colors_list, edgecolor=None)
        plt.legend(labels=df_ETF.columns, fontsize=20)
        plt.xticks(fontsize=20)
        for spine in plt.gca().spines.values():
            spine.set_visible(False)
        plt.yticks([])
        # Add this loop to add the annotations
        for p in ax.patches:
            width = p.get_width()
            height = p.get_height()
            x, y = p.get_xy()
            ax.annotate(f'{height:.0%}', (x + width / 2, y + height), ha='center', fontsize=20)
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.write(f"## ETF Assets Allocation using Critical Line Algorithm")
        st.pyplot()

    if len(moroccan_stocks) != 0:
        df_moroccan_stocks = calculCLA(df_moroccan_stocks)
        result_pct = df_moroccan_stocks.div(df_moroccan_stocks.sum(1), axis=0)
        ax = result_pct.plot(kind='bar', figsize=(20, 6), width=0.6, color=colors_list, edgecolor=None)
        plt.legend(labels=df_moroccan_stocks.columns, fontsize=20)
        plt.xticks(fontsize=20)
        for spine in plt.gca().spines.values():
            spine.set_visible(False)
        plt.yticks([])
        # Add this loop to add the annotations
        for p in ax.patches:
            width = p.get_width()
            height = p.get_height()
            x, y = p.get_xy()
            ax.annotate(f'{height:.0%}', (x + width / 2, y + height), ha='center', fontsize=20)
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.write(f"## Moroccan Assets Allocation using Critical Line Algorithm")
        st.pyplot()
