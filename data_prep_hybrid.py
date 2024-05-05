from eq_data_loader import get_eq_data
import json
import pandas as pd

import numpy as np
import tensorflow as tf

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

data_config = json.load(open('/workspaces/runtime_test/data_config.json', 'r'))

data = get_eq_data(data_config['correlation_thresh'])

#Return calculation
def returnCalculation (database,lag):
    """
    Function to calculate lag return from the input
    -----------------------------------------------
    Input: 
        database (array/pandas.Series) : price data 
        lag (int) : lag for return calculation
    
    Returns: 
        array : 

    """
    dimension=database.shape[0];dif=lag;Out=np.zeros([dimension-dif])
    for i in range(dimension-dif):
        Out[i]=(np.log(database[i+dif])-np.log(database[i]))
    return np.append(np.repeat(np.nan, dif),Out)

#STD Calculation
def sdCalculation (dailyReturns, lagSD):
    """
    Function to calculate lag standard deviation from the input
    -----------------------------------------------
    Input: 
        database (array/pandas.Series) : price data 
        lag (int) : lag for std calculation
    
    Returns: 
        array : 

    """
    dimension=dailyReturns.shape[0]; dif=lagSD; Out=np.zeros([dimension-dif])
    for i in range (dimension-dif):
        Out[i]=np.std(dailyReturns[i:i+lagSD],ddof=1)
    return np.append(np.repeat(np.nan, dif),Out)

def hybrid_transformer_database(data, timestep, xdata, ydata, lag, lagSD, test_size): 
    return_data = pd.DataFrame(index = data.index)
    for col in data.columns: 
        return_data[col + '_lag_return'] = returnCalculation(data[col], lag)
        return_data[col + '_lag_sd'] = sdCalculation(data[col], 5) 
        return_data[col + '_true_return'] = returnCalculation(data[col], 1) 
    return_data.dropna(inplace = True)

    xdata = return_data[[col for col in return_data.columns if 'true' not in col]].copy()
    ydata = return_data[[col for col in return_data.columns if 'true' in col]].copy() 

    scaler = StandardScaler().fit(xdata) 
    scaled_xdata = scaler.transform(xdata) 

    features = scaled_xdata.shape[1] 
    sample = scaled_xdata.shape[0] - timestep + 1 
    output = ydata.shape[1] 

    xdataTrainScaledRNN = np.zeros([sample, timestep, features]) 
    ydataTrainRNN = ydata.iloc[timestep - 1:, :].values 
    for i in range(sample): 
        xdataTrainScaledRNN[i, :, :] = scaled_xdata[i:(timestep + i)]
    
    xtrain, xtest, ytrain, ytest = train_test_split(xdataTrainScaledRNN, ydataTrainRNN)

    return (
        tf.convert_to_tensor(xtrain, np.float32), 
        tf.convert_to_tensor(xtest, np.float32), 
        tf.convert_to_tensor(ytrain, np.float32), 
        tf.convert_to_tensor(ytest, np.float32)
    )
