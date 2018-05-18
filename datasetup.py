
import requests


import numpy as np
import pandas as pd
import pickle
import quandl
import time


from pandas import datetime
from cryptory import Cryptory


__author__ = "Jonathan Rebane extension of work by Guillaume Chevalier"
# - https://github.com/guillaume-chevalier/
__version__ = "2018-03"


def loadCurrency(curr, window_size):
    """
    Return the historical data for the USD or EUR or GBP bitcoin value. Is performed with an API call.
    """
    r = requests.get(
        "http://api.coindesk.com/v1/bpi/historical/close.json?start=2015-08-25&end=2018-04-04&currency={}".format(
            curr
        )
    )
    data = r.json()
    time_to_values = sorted(data["bpi"].items())
    values = [val for key, val in time_to_values]
    kept_values = values #dont remove 
    

    X = []
    Y = []
    for i in range(len(kept_values) - window_size * 2):
        X.append(kept_values[i:i + window_size])
        Y.append(kept_values[i + window_size:i + window_size * 2])

    # To be able to concat on inner dimension later on:
    X = np.expand_dims(X, axis=2)
    Y = np.expand_dims(Y, axis=2)

    return X, Y


def loadCurrencynormal(curr):
    """
    Return the historical data for the USD or EUR or GBP bitcoin value. Is done with an web API call.
    """
    r = requests.get(
        "http://api.coindesk.com/v1/bpi/historical/close.json?start=2015-08-25&end=2018-04-04&currency={}".format(
            curr
        )
    )
    data = r.json()
    time_to_values = sorted(data["bpi"].items())
    values = [val for key, val in time_to_values]

    return values

def loadaltCurrency(altcoin, window_size):
    """
    Return the historical data for altcoins
    """

    kept_values = altcoin

    X = []
    Y = []
    for i in range(len(kept_values) - window_size * 2):
        X.append(kept_values[i:i + window_size])
        Y.append(kept_values[i + window_size:i + window_size * 2])

    # To be able to concat on inner dimension later on:
    X = np.expand_dims(X, axis=2)
    Y = np.expand_dims(Y, axis=2)

    return X, Y


def altData(altcoin, window_size):
    """
    Prepare data windows
    """

    kept_values = altcoin

    X = []
    Y = []
    for i in range(len(kept_values) - window_size * 2):
        X.append(kept_values[i:i + window_size])
        Y.append(kept_values[i + window_size:i + window_size * 2])

    # To be able to concat on inner dimension later on:
    X = np.expand_dims(X, axis=2)
    Y = np.expand_dims(Y, axis=2)

    return X, Y

#preprocessing and collection approach from: https://blog.patricktriest.com/analyzing-cryptocurrencies-python/

def get_quandl_data(quandl_id):
    '''Download and cache Quandl dataseries'''
    cache_path = '{}.pkl'.format(quandl_id).replace('/','-')
    try:
        f = open(cache_path, 'rb')
        df = pickle.load(f)   
        print('Loaded {} from cache'.format(quandl_id))
    except (OSError, IOError) as e:
        print('Downloading {} from Quandl'.format(quandl_id))
        df = quandl.get(quandl_id, returns="pandas")
        df.to_pickle(cache_path)
        print('Cached {} at {}'.format(quandl_id, cache_path))
    return df

btc_usd_price_kraken = get_quandl_data('BCHARTS/KRAKENUSD')


exchanges = ['COINBASE','BITSTAMP','ITBIT']

exchange_data = {}

exchange_data['KRAKEN'] = btc_usd_price_kraken

def merge_dfs_on_column(dataframes, labels, col):
    '''Merge a single column of each dataframe into a new combined dataframe'''
    series_dict = {}
    for index in range(len(dataframes)):
        series_dict[labels[index]] = dataframes[index][col]
        
    return pd.DataFrame(series_dict)



for exchange in exchanges:
    exchange_code = 'BCHARTS/{}USD'.format(exchange)
    btc_exchange_df = get_quandl_data(exchange_code)
    exchange_data[exchange] = btc_exchange_df
    
btc_usd_datasets = merge_dfs_on_column(list(exchange_data.values()), list(exchange_data.keys()), 'Weighted Price')
btc_usd_datasets.replace(0, np.nan, inplace=True)

btc_usd_datasets['avg_btc_price_usd'] = btc_usd_datasets.mean(axis=1)

def get_json_data(json_url, cache_path):
    '''Download and cache JSON data, return as a dataframe.'''
    try:        
        f = open(cache_path, 'rb')
        df = pickle.load(f)   
        print('Loaded {} from cache'.format(json_url))
    except (OSError, IOError) as e:
        print('Downloading {}'.format(json_url))
        df = pd.read_json(json_url)
        df.to_pickle(cache_path)
        print('Cached {} at {}'.format(json_url, cache_path))
    return df

base_polo_url = 'https://poloniex.com/public?command=returnChartData&currencyPair={}&start={}&end={}&period={}'
start_date = datetime.strptime('2015-08-25', '%Y-%m-%d') # get data from the start of 2016
end_date = datetime.strptime('2018-04-05', '%Y-%m-%d') # up until today
pediod = 86400 # pull daily data (86,400 seconds per day)

def get_crypto_data(poloniex_pair):
    '''Retrieve cryptocurrency data from poloniex'''
    json_url = base_polo_url.format(poloniex_pair, time.mktime(start_date.timetuple()), time.mktime(end_date.timetuple()), pediod)
    data_df = get_json_data(json_url, poloniex_pair)
    data_df = data_df.set_index('date')
    return data_df

altcoins = ['ETH','LTC','XRP','ETC','STR','DASH','SC','XMR','XEM']

altcoin_data = {}
for altcoin in altcoins:
    coinpair = 'BTC_{}'.format(altcoin)
    crypto_price_df = get_crypto_data(coinpair)
    altcoin_data[altcoin] = crypto_price_df


for altcoin in altcoin_data.keys():
    altcoin_data[altcoin]['price_usd'] =  altcoin_data[altcoin]['weightedAverage'] * btc_usd_datasets['avg_btc_price_usd']


# Merge USD price of each altcoin into single dataframe 
combined_df = merge_dfs_on_column(list(altcoin_data.values()), list(altcoin_data.keys()), 'price_usd')

# Add BTC price to the dataframe
combined_df['BTC'] = btc_usd_datasets['avg_btc_price_usd']


def singleoutputnormalizeX(X, Y=None):
    """
    Normalise X and Y according to the mean and standard deviation of the X values only.
    """
    # # It would be possible to normalize with last rather than mean, such as:
    # lasts = np.expand_dims(X[:, -1, :], axis=1)
    # assert (lasts[:, :] == X[:, -1, :]).all(), "{}, {}, {}. {}".format(lasts[:, :].shape, X[:, -1, :].shape, lasts[:, :], X[:, -1, :])
    mean = np.expand_dims(np.average(X, axis=1) + 0.00001, axis=1)
    stddev = np.expand_dims(np.std(X, axis=1) + 0.00001, axis=1)
    # print (mean.shape, stddev.shape)
    # print (X.shape, Y.shape)
    X = X - mean
    X = X / (2.5 * stddev)
    if Y is not None:
#        assert Y.shape == X.shape, (Y.shape, X.shape)
        Y_mean = np.expand_dims(mean[:,:,0], axis=1)
        Y_std = (2.5 * np.expand_dims(stddev[:,:,0], axis=1))
        Y = Y - Y_mean
        Y = Y / Y_std
        return X, Y, Y_mean, Y_std
    return X


def fetch_batch_size_random(X, Y, batch_size):
    """
    Returns randomly an aligned batch_size of X and Y among all examples.
    The external dimension of X and Y must be the batch size (eg: 1 column = 1 example).
    X and Y can be N-dimensional.
    """
#   assert X.shape == Y.shape, (X.shape, Y.shape)
    idxes = np.random.randint(X.shape[0], size=batch_size)
    X_out = np.array(X[idxes]).transpose((1, 0, 2))
    Y_out = np.array(Y[idxes]).transpose((1, 0, 2))
    
    mean = np.expand_dims(np.average(X_out, axis=1) + 0.00001, axis=1)
    stddev = np.expand_dims(np.std(X_out, axis=1) + 0.00001, axis=1)
    
    Ymean = np.expand_dims(mean[:,:,0], axis=1)
    Ystd = (2.5 * np.expand_dims(stddev[:,:,0], axis=1))
    
    return X_out, Y_out


def fetch_all(X, Y):
    """
    Returns all alligned examples of X and Y
    """
#   assert X.shape == Y.shape, (X.shape, Y.shape)
    idxes = range(X.shape[0])
    X_out = np.array(X[idxes]).transpose((1, 0, 2))
    Y_out = np.array(Y[idxes]).transpose((1, 0, 2))
    
    mean = np.expand_dims(np.average(X_out, axis=1) + 0.00001, axis=1)
    stddev = np.expand_dims(np.std(X_out, axis=1) + 0.00001, axis=1)
    
    Ymean = np.expand_dims(mean[:,:,0], axis=1)
    Ystd = (2.5 * np.expand_dims(stddev[:,:,0], axis=1))
    
    return X_out, Y_out



X_train = []
Y_train = []
X_test = []
Y_test = []

def get_mean():
    global Y_mean
    return Y_mean

def get_std():
    global Y_std
    return Y_std

#####GET GOOGLE TRENDS#####
my_cryptory = Cryptory(from_date="2015-08-25", to_date="2018-04-04")
trend1=my_cryptory.get_google_trends(kw_list=['bitcoin'])
trend2=my_cryptory.get_google_trends(kw_list=['ethereum'])


def generatedata_v1(isTrain, batch_size):
    """
    Return data regarding cyrptocurrencies.

    """
    # window for encoder and decoder's predictions.
    seq_length = 40

    global Y_train
    global X_train
    global X_test
    global Y_test
    global Y_mean
    global Y_std
    # First load, with memoization:
    if len(Y_test) == 0:
        #get more data
        
        X_google, Y_google = altData(trend1["bitcoin"].tolist()[::-1], window_size=seq_length)
        X_google2, Y_google2 = altData(trend2["ethereum"].tolist()[::-1], window_size=seq_length)
        X_usd, Y_usd = loadCurrency("USD", window_size=seq_length) # or loadaltCurrency(combined_df["BTC"].tolist(), window_size=seq_length)
        X_eur, Y_eur = loadCurrency("EUR", window_size=seq_length)
        X_gbp, Y_gbp = loadCurrency("GBP", window_size=seq_length)
        X_dash, Y_dash = altData(combined_df["DASH"].tolist(), window_size=seq_length)
        X_etc, Y_etc = altData(combined_df["ETC"].tolist(), window_size=seq_length)
        X_eth, Y_eth = altData(combined_df["ETH"].tolist(), window_size=seq_length)
        X_ltc, Y_ltc = altData(combined_df["LTC"].tolist(), window_size=seq_length)
        X_sc, Y_sc = altData(combined_df["SC"].tolist(), window_size=seq_length)
        X_str, Y_str = altData(combined_df["STR"].tolist(), window_size=seq_length)
        X_xem, Y_xem = altData(combined_df["XEM"].tolist(), window_size=seq_length)
        X_xmr, Y_xmr = altData(combined_df["XMR"].tolist(), window_size=seq_length)
        X_xrp, Y_xrp = altData(combined_df["XRP"].tolist(), window_size=seq_length)
        # All data. choose input configuration:
        #X = np.concatenate((X_usd, X_eur, X_gbp), axis=2) #bitcoin only
        #X = np.concatenate((X_usd, X_eur, X_gbp, X_dash, X_eth, X_ltc, X_sc, X_str, X_xem, X_xmr, X_xrp), axis=2) #with altcoin
        X = np.concatenate((X_usd, X_eur, X_gbp, X_dash, X_ltc, X_sc, X_str, X_xem, X_xmr, X_xrp, X_google, X_google2), axis=2) #with altcoin + google trends
        

        Y = Y_usd
        X, Y, Y_mean, Y_std = singleoutputnormalizeX(X, Y)

        # Split to training and test windows:
        X_train = X[:int(len(X) * 0.8)]
        Y_train = Y[:int(len(Y) * 0.8)]
        X_test = X[int(len(X) * 0.8):]
        Y_test = Y[int(len(Y) * 0.8):]

    if isTrain:
        return fetch_batch_size_random(X_train, Y_train, batch_size)
    else:
        return fetch_batch_size_random(X_test,  Y_test,  batch_size)
    
    
def generatedata_v2(isTrain):
    """
    Return all prediction windows
    """
    return fetch_all(X_test,  Y_test)
    


    

    
    

    
    
    
    
    
    
    
    
    
    
    

    
    
    
#########################