
import pandas as pd 
import yfinance as yf
import json


config = json.load(open('/workspaces/runtime_test/data_config.json', 'r'))
nf_50_tickers_df = pd.read_csv('/workspaces/runtime_test/nifty_50_metadata.csv')[['Industry', 'Symbol']]
nf_50_tickers_df.Symbol = nf_50_tickers_df.Symbol + '.NS'
nf_50_tickers = nf_50_tickers_df.Symbol.tolist() 

data = yf.download(
    nf_50_tickers, 
    start = config['data_start_date'],
    end = config['data_end_date']
)['Adj Close'].dropna(axis = 1)

corr = data.corr() 
corr_lt_05 = corr[corr < 0.45]
corr_lt_05.dropna(thresh = 6, axis = 0, inplace = True)
corr_lt_05.dropna(thresh = 6, axis = 1, inplace = True)
print(corr_lt_05.shape)

print(nf_50_tickers_df[nf_50_tickers_df.Symbol.isin(corr_lt_05.columns)].groupby('Industry').agg(
    stock_count = ('Symbol', 'nunique')
))