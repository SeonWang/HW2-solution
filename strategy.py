from strategyData import *
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
import seaborn as sns
import copy
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
import statsmodels.api as sm
from pickle import dump
from pickle import load
import pandas as pd
import numpy as np
import pandas_datareader as pdr

# zhong yi's strategy
def strategy (size, md):
    df = pdr.DataReader('IVV', data_source='yahoo', start='2020-04-01')
    bond_10 = pdr.DataReader('^TNX',data_source='yahoo', start='2020-04-01')
    bond_30 = pdr.DataReader('^TYX',data_source='yahoo', start='2020-04-01')
    bond_5 = pdr.DataReader('^FVX',data_source='yahoo', start='2020-04-01')
    final = len(df)
    eia = pdr.DataReader('EIA.F',data_source='yahoo',start='2020-04-01')
    df['IVV'] = (df['Close'] - df['Open']) / df['Open']
    df['mvg30'] = df['IVV'].rolling(window=size).mean()
    df['std30'] = df['IVV'].rolling(window=size).std()
    bond_10['bond10'] = (bond_10['Close'] - bond_10['Open']) / bond_10['Open']
    bond_5['bond5'] = (bond_5['Close'] - bond_5['Open']) / bond_5['Open']
    bond_30['bond30'] = (bond_30['Close'] - bond_30['Open']) / bond_30['Open']
    ivv_response = []
    for i in range(1, len(df)):
        ivv_response.append(int(df["IVV"][i] > 0))
    ivv_response.append(np.nan)
    df['Response'] = ivv_response
    eia_return = []
    eia_return.append(np.nan)
    for i in range(1, len(eia)):
        eia_return.append((eia['Close'][i] - eia['Close'][i - 1]) / eia['Close'][i - 1])
    eia['eia'] = eia_return
    kp_df = ['Date', 'IVV', 'mvg30', 'std30', 'Response','Close','Open']
    dp_df = [col for col in df.columns if col not in kp_df]
    df.drop(labels=dp_df, axis=1, inplace=True)
    kp_5 = ['Date', 'bond5']
    dp_5 = [col for col in bond_5.columns if col not in kp_5]
    bond_5.drop(labels=dp_5, axis=1, inplace=True)
    kp_10 = ['Date', 'bond10']
    dp_10 = [col for col in bond_10.columns if col not in kp_10]
    bond_10.drop(labels=dp_10, axis=1, inplace=True)
    kp_30 = ['Date', 'bond30']
    dp_30 = [col for col in bond_30.columns if col not in kp_30]
    bond_30.drop(labels=dp_30, axis=1, inplace=True)
    kp_e = ['Date', 'eia']
    dp_e = [col for col in eia.columns if col not in kp_e]
    eia.drop(labels=dp_e, axis=1, inplace=True)
    model_data = df.join(bond_5, how='outer')
    model_data = model_data.join(bond_10, how='outer')
    model_data = model_data.join(bond_30, how='outer')
    model_data = model_data.join(eia, how='outer')
    test_data = model_data[-1:]
    model_data = model_data.dropna(axis=0)
    Y = model_data['Response']
    X = model_data.loc[:, model_data.columns!='Response']
    validation_size = 0.2
    seed = 3
    X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_size, random_state=seed)
    if md =='Decision Tree':
        model = DecisionTreeClassifier()
    if md =='Loglinear':
        model = LogisticRegression()
    if md =='KNN':
        model = KNeighborsClassifier()
    model.fit(X_train, Y_train)
    filename = 'finalized_model.sav'
    dump(model, open(filename, 'wb'))
    return model_data, test_data

# yuzhe's strategy
def logistic_strategy(df, start, train_window):
    Y_hat = []
    df['sum3'] = df['ivv_signal'].rolling(window=3).sum()
    df = df.dropna()
    for i in range(train_window+1, len(df.index)+1):
        Y = np.array(df['ivv_signal'].iloc[i - train_window: i])
        X = np.array(df[['ROE', 'TR', 'inflation', 'sum3']].iloc[i - train_window - 1: i-1])
        logreg = LogisticRegression(max_iter=2000)
        logreg.fit(X, Y)
        Y_hat.append(logreg.predict(
            np.array(df[['ROE', 'TR', 'inflation', 'sum3']].iloc[i - 1: i]))[-1])
    df = df.iloc[train_window+1:]
    df['actn'] = Y_hat[:-1]

    start = pd.to_datetime(start)
    for day in df.index:
        real_start_index = day
        if day.__ge__(start):
            break
    end = df.index[len(df.index) - 1]
    df = df.loc[real_start_index:, :]

    df.to_csv('logistic_results.csv')

    return df