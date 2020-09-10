import pickle
import operator
import time
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta


def load_data(filename):
    SESSION_LENGTH = 30 * 60
    data = pd.read_csv(filename, sep=',', header=0, usecols=[0, 1, 2, 3],
                       dtype={0: np.int64, 1: np.int32, 2: str, 3: np.int32})
    data.columns = ['Time', 'UserId', 'Type', 'ItemId']
    data['Time'] = (data.Time / 1000).astype(int)
    data.sort_values(['UserId', 'Time'], ascending=True, inplace=True)
    data['TimeTmp'] = pd.to_datetime(data.Time, unit='s')
    data.sort_values(['UserId', 'TimeTmp'], ascending=True, inplace=True)

    data['TimeShift'] = data['TimeTmp'].shift(1)
    data['TimeDiff'] = (data['TimeTmp'] - data['TimeShift']).dt.total_seconds().abs()
    data['SessionIdTmp'] = (data['TimeDiff'] > SESSION_LENGTH).astype(int)
    data['SessionId'] = data['SessionIdTmp'].cumsum(skipna=False)
    del data['SessionIdTmp'], data['TimeShift'], data['TimeDiff']

    data.sort_values(['SessionId', 'Time'], ascending=True, inplace=True)
    cart = data[data.Type == 'addtocart']
    data = data[data.Type == 'view']
    del data['Type']

    print(data.Time.min())
    print(data.Time.max())
    data_start = datetime.fromtimestamp(data.Time.min(), timezone.utc)
    data_end = datetime.fromtimestamp(data.Time.max(), timezone.utc)
    del data['TimeTmp']
    print('Loaded data set')
    print('Events: ', len(data))
    print('Sessions: ', data.SessionId.nunique())
    print('Items: ', data.ItemId.nunique())
    print('Span: {} / {}'.format(data_start.date().isoformat(), data_end.date().isoformat()))

    return data, cart


data, _ = load_data('./events.csv')
data.to_csv('./event1.csv', index=False)
