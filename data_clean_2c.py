import numpy as np
import pandas as pd
import datetime
import get_data
from joblib import Parallel, delayed


# adding lags to price and volume data as features
def add_lag(df, level, n_shifts, lag1_present = False, method = "Method1"):

    if level == 5:

        col_arr = ['ASK_PRICE1', 'ASK_PRICE2', 'ASK_PRICE3', 'ASK_PRICE4', 'ASK_PRICE5',
               'BID_PRICE1', 'BID_PRICE2', 'BID_PRICE3', 'BID_PRICE4', 'BID_PRICE5',
               'ASK_SIZE1', 'ASK_SIZE2', 'ASK_SIZE3', 'ASK_SIZE4', 'ASK_SIZE5',
               'BID_SIZE1', 'BID_SIZE2', 'BID_SIZE3', 'BID_SIZE4', 'BID_SIZE5']

    elif level == 1:
        col_arr = ['ASK_PRICE', 'BID_PRICE', 'ASK_SIZE', 'BID_SIZE']

    if lag1_present == True:
        start = 2

    else:

        start = 1

    if method == "Method1":
        lag_dict = {}
        for j in range(start, n_shifts+1):
            for col in col_arr:
                key = col + '_LAG' + str(j)
                lag_dict[key] = df[col].shift(j)
        df_lag = pd.DataFrame.from_dict(lag_dict)
        df = pd.concat([df, df_lag], axis=1)
        df = df.dropna(axis=0).reset_index(drop=True)

    elif method == "Method2":

        for j in range(start, n_shifts+1):
            for i in range(len(col_arr)):
                col = col_arr[i]
                col_new = col + '_LAG' + str(j)
                df[col_new] = df[col].shift(j)
        df = df.dropna(axis=0).reset_index(drop=True)

    return df


def gety2_parallel(data, timestamp, midprice, start, level = 5):
    if level == 5:
        for index in range(start+1, len(data)):
            if data["MID_PRICE1"].iloc[index] != midprice:
                return index
    elif level == 1:
        for index in range(start+1, len(data)):
            if data["MID_PRICE"].iloc[index] != midprice:
                return index
    return start


# basic function to construct y labels for data
def gety2c(data, tdelta, level=5):

    # get the corresponding y
    N_samples = data.shape[0]
    y_idx = [0]*N_samples
    y_data = [0.0]*N_samples
    y_label = [[0.0]*2]*N_samples

    if level == 5:

        data["MID_PRICE1"] = ( data["BID_PRICE1"] + data["ASK_PRICE1"] )* 0.5

        i=0
        while i < N_samples:
            y_idx[i] = gety2_parallel(data,
                        data['#TIMESTAMP'].iloc[i],
                        data["MID_PRICE1"].iloc[i],
                        start=i,
                        level = 5)

            if y_idx[i] != i:
                for j in range(i+1, y_idx[i]):
                    y_idx[j] = y_idx[i]
                i = y_idx[i]
            else:
                i = i + 1

    elif level == 1:

        data["MID_PRICE"] = ( data["BID_PRICE"] + data["ASK_PRICE"] )* 0.5
        print("start y idx")
        i=0
        while i < N_samples:
            y_idx[i] = gety2_parallel(data,
                        data['#TIMESTAMP'].iloc[i],
                        data["MID_PRICE"].iloc[i],
                        start=i,
                        level = 5)

            if y_idx[i] != i:
                for j in range(i+1, y_idx[i]):
                    y_idx[j] = y_idx[i]
                i = y_idx[i]
            else:
                i = i + 1

        print("end y idx")

    for i in range(N_samples):

        if level==5:
            y_data[i] = data["MID_PRICE1"].iloc[y_idx[i]] - data["MID_PRICE1"].iloc[i]
        else:
            y_data[i] = data["MID_PRICE"].iloc[y_idx[i]] - data["MID_PRICE"].iloc[i]

        if y_data[i] > 0.0:
            y_label[i] = [1.0, 0.0]
        elif y_data[i] < 0.0:
            y_label[i] = [0.0, 1.0]
        elif y_data[i] == 0.0:
            y_label[i] = np.nan

    return y_label


# basic data clean and feature engineering function
def data_clean1(data, n_lags = 6, level=5):

    data['#TIMESTAMP'] = pd.to_datetime(data['#TIMESTAMP'], format="%Y-%m-%d %H:%M:%S.%f")

    if level==1:

        drop_cols = ['SEQ_NUM', 'SIZE', 'ASK_SIZE_IMPLIED', 'BID_SIZE_IMPLIED', 
                    'PRICE', 'ASK_SIZE_OUTRIGHT', 'BID_SIZE_OUTRIGHT', 'SYMBOL_NAME']

        # Dropping redundant and unnecessary columns
        data = data.drop(drop_cols, axis=1)

        data = data[data['BID_PRICE'] < data['ASK_PRICE']]

    elif level == 5:

        drop_cols = ['SIZE', 'SYMBOL_NAME', 'PRICE']

        # Dropping redundant and unnecessary columns
        data = data.drop('drop_cols', axis = 1)

        data = data[data['BID_PRICE1'] < data['ASK_PRICE1']]
        data = data[data['BID_PRICE2'] < data['ASK_PRICE2']]
        data = data[data['BID_PRICE3'] < data['ASK_PRICE3']]
        data = data[data['BID_PRICE4'] < data['ASK_PRICE4']]
        data = data[data['BID_PRICE5'] < data['ASK_PRICE5']].reset_index(drop=True)

    # Drop completely duplicate rows
    data = data.drop_duplicates().reset_index(drop=True)

    # Dropping duplicate timestamps, only keeping the last
    data = data.drop_duplicates(subset=['#TIMESTAMP'], keep='last').reset_index(drop=True)

    # Dropping rows with NaN values
    data = data.dropna(axis=0).reset_index(drop=True)

    # convert timestamp to a feature
    data['#TIMENORM'] = (data['#TIMESTAMP'] - data['#TIMESTAMP'].min()) / datetime.timedelta(seconds=1) / 3600.0

    # time difference between successive snapshots of the orderbook
    data['TIMEDELTA'] = np.array(data['#TIMENORM'].diff())

    # Dropping rows with NaN values
    data = data.dropna(axis=0).reset_index(drop=True)

    # Removing #TIMENORM column due to non-stationarity
    data = data.drop('#TIMENORM', axis=1)

    # adding lagged variable features to dataset
    data = add_lag(data, level=level, n_shifts=n_lags, lag1_present = True, method = "Method1")

    # Dropping rows with NaN values
    data = data.dropna(axis=0).reset_index(drop=True)

    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    days_no = [0, 1, 2, 3, 4, 5, 6]
    days_dict = dict(zip(days_no, days))

    # code to encode days as features
    dayofweek = data["#TIMESTAMP"].dt.dayofweek
    dayofweek_dummies = pd.get_dummies(pd.DataFrame({'day': [days_dict[val] for val in dayofweek],
                                                    "#TIMESTAMP" : data["#TIMESTAMP"]}))
    data = pd.concat([data, dayofweek_dummies], axis = 1, join = 'outer')

    # remove duplicated columns
    data = data.loc[:, ~data.columns.duplicated()]

    if data.isnull().values.any() == True:
        print("NaN exists! Check!")
    else:
        print("data clean - no NaN")

    return data


# function calls data_clean1 to clean data 
def clean_process(fname, directory, n_lags, data_clean_flag = 1, level=5):

    if level==5:
        n=3
    else:
        n=2

    if data_clean_flag == 1:
        data = pd.read_csv(directory + "raw_data/" + fname +".csv", skiprows=n)
        dc = data_clean1(data, n_lags, level = level)
    else:
        dc = pd.read_csv(directory + "cleaned_data/" + fname +"_clean.csv")
        dc['#TIMESTAMP'] = pd.to_datetime(dc['#TIMESTAMP'], format="%Y-%m-%d %H:%M:%S.%f")

    # construct y from cleaned data
    y_data = gety2c(dc, tdelta=1, level=level)

    dc["y"] = y_data
    # Dropping rows with NaN values
    dc = dc.dropna(axis=0).reset_index(drop=True)
    # Making a copy of y
    y_data = dc["y"].copy(deep=True)
    # Removing y from dc
    dc = dc.drop("y", axis=1)

    if data_clean_flag == 1:
        dc.to_csv(directory + "cleaned_data_2c/" + fname +"_clean.csv")

    with open(directory + "cleaned_data_2c/" + "y_"+ fname + "_data.txt", 'w') as f:
        for item in y_data:
            f.write("%s\n" % item)

    return 0


directory = "D:/Dropbox/Applied_Finance_Project/codebase/Data/5L/"
data_clean_flag = 1

fn1 = "ZC4H_20131205_20140304"
clean_process(fn1, directory=directory, data_clean_flag=1, level=5)
print(fn1 + " done")