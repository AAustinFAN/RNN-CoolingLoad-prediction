import time

import pandas as pd
import torch
import datetime
import numpy as np

features = ['coolingLoad', 'temperature']
seq_lenth = 5
mean_list =[] #the mean an std of each feature
std_list = []


def create_X_Y(np_array,sequence_length, conti_list):
    X,Y = [],[]
    for i in conti_list:
        X.append(np_array[i:i+sequence_length])
        Y.append(np_array[i+sequence_length][0])
    X = np.array(X)
    Y = np.array(Y).reshape(len(Y),1)
    return X,Y

def str_data_to_num(str_data):
    # 1. format time
    strptime = time.strptime(str_data,"%Y/%m/%d %H:%M")
    return strptime.tm_hour

def if_conti(df):
    #1.find the continue time seties
    conti_list = []
    for i in range(len(df)-seq_lenth-1):
        t1 = str_data_to_num(df['time'][i])
        t2 = str_data_to_num(df['time'][i+seq_lenth])
        if t2-t1 == seq_lenth:
            conti_list.append(i)
    return conti_list

def normalization(data):
    df_normal = pd.DataFrame()
    for feature in features:
        df = data[feature]

        df_numpy_mean = np.mean(df)
        df_numpy_std = np.std(df)
        df_numpy = (df - df_numpy_mean) / df_numpy_std

        df_normal[feature] = df_numpy

        mean_list.append(df_numpy_mean)
        std_list.append(df_numpy_std)
    return df_normal


def readdata_train():
    # 1.load the csv data
    # data = pd.read_excel('data/w2.xlsx',usecols=['coolingLoad','temperature'])
    data = pd.read_csv('data/CP4.csv')
    data = data.loc[data['cop']>0,:]
    # 2.find the continue time list
    conti_list = if_conti(data)

    data = data[features]
    # 3. data normalization
    df_numpy = normalization(data)
    df_numpy = df_numpy.values
    # 4.generate the feature and label
    X,Y = create_X_Y(df_numpy,seq_lenth,conti_list)
    return X,Y

def readdata_test(mean_list,std_list):
    # 1.load the csv data
    # data = pd.read_excel('data/w2.xlsx',usecols=['coolingLoad','temperature'])
    data = pd.read_csv('data/CP4.csv')
    data = data.loc[data['cop']>0,:]
    # 2.find the continue time list
    conti_list = if_conti(data)
    data = data[features]
    # 3. data normalization

    df_normal = pd.DataFrame()
    for i in range(len(features)):
        feature = features[i]
        df = data[feature]
        print(df)
        df_numpy = (df - mean_list[i]) / std_list[i]
        print(df_numpy)
        df_normal[feature] = df_numpy

    df_numpy = df_normal.values
    # 4.generate the feature and label
    X,Y = create_X_Y(df_numpy,seq_lenth,conti_list)
    print(X,Y)
    return X, Y

datax, datay= readdata_train()
x,y = readdata_test(mean_list,std_list)
