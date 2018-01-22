import pandas as pd
import datetime


def loadData(filename):
    data = pd.read_csv('coinbaseEUR.csv', index_col=0, header=None, error_bad_lines=False)
    data.index = pd.to_datetime((data.index.values*1e9).astype(int))
    return data
