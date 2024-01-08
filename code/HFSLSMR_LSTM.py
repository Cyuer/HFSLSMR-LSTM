import math
import os

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import deque
from sympy import integrate, exp, sin, log, oo, pi,symbols
from keras.metrics import RootMeanSquaredError,r_square
from keras.models import Sequential
from keras import optimizers
from keras.layers import LSTM,Dense,Dropout,GRU
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
scaler = MinMaxScaler()
def main(data):
    # Read the file, set the index to the date, and sort it
    df = pd.read_csv(f'paper_data/index/{data}', index_col=0, parse_dates=True)
    df.sort_index(inplace=True)
    df = df.fillna(0,axis=1)
    #Set the prediction to the pre_day ahead and let the closing price move up the pre_day, so that the features and labels correspond to each other, for ease of calculation
    pre_day = 11
    df['label'] = df['label'].shift(-pre_day)
    # Deletion of null values for the last few days after the upward shift
    df = df.iloc[:-pre_day,:]
    fit(df)


def weight(n, k):
    w = [i for i in range(1, k * n + 1, k)]
    w = np.array(w)
    weights = w / sum(w)
    return weights

def fit(df):
    X = df.iloc[:, :-1].apply(lambda x: (x - x.min()) / (x.max()-x.min() + 1e-7), axis=0)
    features = X.columns
    y = df['label'].values[:-window+1]
    y = scaler.fit_transform(y.reshape(-1, 1))
    pred_sub = pd.DataFrame(np.zeros((int(len(y) * 0.7), num_models), dtype=float))
    for k in range(num_models):
        print(f'--------For the {k+1}th layer, I chose the feature：',list(features),f'length：{len(features)}',sep='\n')
        sub_features.append(features)
        x = process_data(X.loc[:, features].values)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3,shuffle=True)
        split_ = int(len(y) * 0.7)
        x__test, y__test = x[split_:, :, :], y[split_:]
        N = len(x_train)
        model_k = train_submodel(x_train, y_train, x__test, y__test)
        ensemble.append(model_k)
        if k + 1 == num_models:
            break

        pred_k = predict_sub(model_k, x_train)
        pred_sub.iloc[:, k] = pred_k

        if s == 0:
            pred_ensemble = pred_sub.iloc[:, : k + 1].mean(axis=1)
            w = np.array([1/(k+1) for i in range(0,k+1)])
        else:
            w = weight(k+1,s)
            for i in range(len(w)):
                pred_sub.iloc[:,i] = pred_sub.iloc[:,i]*w[i]
            pred_ensemble = pred_sub.iloc[:, : k + 1].sum(axis=1)
            # end
        loss_values = pd.Series(get_loss(y_train, pred_ensemble.values))
        if enable_fs:
            features = feature_selection(X,y,y_train,N, loss_values, features,w)

def process_data(X):
    # Setting the sliding window
    que = deque(maxlen=window) #Setting the queue with size window
    x = []
    for i in X:
        que.append(i)
        if len(que) == window:
            x.append(list(que))
    #print(len(x),len(y))

    x = np.array(x)
    return x

def train_submodel(x_train, y_train, x__test, y__test):
    global minloss
    model = Sequential()
    model.add(LSTM(64, input_shape=x_train.shape[1:], activation='relu', return_sequences=False))
    model.add(Dropout(0.1))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss=loss)
    model.fit(x_train, y_train, batch_size=32, epochs=50, validation_split=0.1,verbose=0,shuffle=False)
    y_pre = model.predict(x__test)
    # y__test_ = scaler.fit_transform(y__test.reshape(-1, 1))
    mse = mean_squared_error(y__test, y_pre[:, 0])
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(y__test, y_pre[:, 0])
    mape = mean_absolute_percentage_error(y__test, y_pre[:, 0])
    r2 = r2_score(y__test, y_pre[:, 0])

    print('mse：----', mse)
    if mse < minloss:
        y_pre = scaler.inverse_transform(y_pre)
        y__test = scaler.inverse_transform(y__test)
        measure[0] = [mse,rmse, mae,mape, r2]
        true_pre[0] = list(y__test[:, 0])
        true_pre[1] = list(y_pre[:, 0])
        minloss = mse
    return model

def predict_sub(submodel, x_train):
    pred_sub = pd.Series(submodel.predict(x_train,verbose=0, steps=None).squeeze())
    return pred_sub


def feature_selection(X,y,y_train,N,loss_values, features,w):

    F = len(features)
    E = pd.DataFrame({"E_value": np.zeros(F, dtype=float)})
    M = len(ensemble)
    # shuffle specific columns and calculate g-value for each feature
    X_tmp = X.copy()
    for i_f, feat in enumerate(features):
        X_tmp.loc[:, feat] = np.random.permutation(X_tmp.loc[:, feat].values)
        pred = pd.Series(np.zeros(N))
        for i_s, submodel in enumerate(ensemble):
            x = process_data(X_tmp.loc[:, sub_features[i_s]].values)
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

            pred += (
                    pd.Series(
                        submodel.predict(x_train, verbose=0, steps=None).squeeze()
                    )
                    * w[i_s]  # * w[i_s] 或 / M
            )
        loss_feat = get_loss(y_train, pred.values)
        E.loc[i_f, "E_value"] = np.mean(loss_feat - loss_values) / (np.std(loss_feat - loss_values) + 1e-7)
        X_tmp.loc[:, feat] = X.loc[:, feat].copy()

    # one column in train features is all-nan # if g['g_value'].isna().any()
    E["E_value"].replace(np.nan, 0, inplace=True)
    # divide features into bins_fs bins
    E["bins"] = pd.cut(E["E_value"], bins_fs)
    # randomly sample features from bins to construct the new features
    res_feat = []
    sorted_bins = sorted(E["bins"].unique(), reverse=True)
    # print('sorted_bins :',sorted_bins)
    for i_b, b in enumerate(sorted_bins):
        b_feat = features[E["bins"] == b]
        num_feat = int(np.ceil(sample_ratios[i_b] * len(b_feat)))
        res_feat = res_feat + np.random.choice(b_feat, size=num_feat, replace=False).tolist()
    return pd.Index(set(res_feat))

def get_loss(label, pred):
    if loss == "mse":
        return (np.squeeze(label) - pred) ** 2
    elif loss == 'CrossEntropy':
        return integrate()
    else:
        raise ValueError("not implemented yet")



if __name__ == '__main__':
    window = 10
    loss = "mse"
    sample_ratios = [1, 0.9, 0.9, 0.8, 0.7]
    bins_fs = 5
    enable_fs = True
    sub_features = []
    ensemble = []
    num_models = 6
    minloss = 20
    s = 0
    data_type = 'index'
    measure = [[]]
    true_pre = [[],[]]
    main("SSE.csv")
    print('True:',*true_pre[0],sep=' ')
    print('Pre:', *true_pre[1], sep=' ')
    print('mse,rmse,mae,mape,r2:',*measure[0],sep=' ')
