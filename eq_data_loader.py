
import pandas as pd 
import yfinance as yf
import json

def get_eq_data(corr_thresh):
    """
    A function that select the equities out of nifty50 based on these criteria: 
        - equity is traded during selected date range and no blank
        - all selected equities have mutual correlation less than a threshold

    return: DataFrame 
    """
    data_config = json.load(open('/workspaces/runtime_test/data_config.json', 'r'))
    nf_50_tickers_df = pd.read_csv('/workspaces/runtime_test/nifty_50_metadata.csv')[['Industry', 'Symbol']]
    nf_50_tickers_df.Symbol = nf_50_tickers_df.Symbol + '.NS'
    nf_50_tickers = nf_50_tickers_df.Symbol.tolist() 

    data = yf.download(
        nf_50_tickers, 
        start = data_config['data_start_date'],
        end = data_config['data_end_date']
    )['Adj Close'].dropna(axis = 1)

    corr = data.corr() 
    corr_lt_thresh = corr[corr < corr_thresh]
    corr_lt_thresh.dropna(thresh = 6, axis = 0, inplace = True)
    corr_lt_thresh.dropna(thresh = 6, axis = 1, inplace = True)

    return data[corr_lt_thresh.columns]

if __name__ == '__main__': 
    pass