import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
import seaborn as sns
#from filterpy.kalman import KalmanFilter
#from filterpy.common import Q_discrete_white_noise
from tqdm import tqdm

def preprocessing1(input_path):
  data = pd.read_csv('\\Users\\ineeji\\Desktop\\새 폴더\\Ineeji\\datas\\데이터합본_파생변수 제거.csv')

  data['year'] = data['Unnamed: 0'].apply(lambda x : x.split()[0].split('-')[0])
  data['month'] = data['Unnamed: 0'].apply(lambda x : x.split()[0].split('-')[1])
  data['date'] = data['Unnamed: 0'].apply(lambda x : x.split()[0].split('-')[2])
  data['hour'] = data['Unnamed: 0'].apply(lambda x : x.split()[1].split(' ')[0].split(':')[0])

  data['year'] = data['year'].astype('int')
  data['month'] = data['month'].astype('int')
  data['date'] = data['date'].astype('int')
  data['hour'] = data['hour'].astype('int')
  return data

def Tree_sigma(df):
  lower_out = df['DSL D-95'].mean() - df['DSL D-95'].std()*3
  upper_out = df['DSL D-95'].mean() + df['DSL D-95'].std()*3
  df = df[(df['DSL D-95'] > lower_out) & (df['DSL D-95'] < upper_out)]
  return df

def drop_List(df,train,test, n):
  dl = df.corr()['DSL D-95'].map(abs).sort_values(ascending = False)[10:]
  #dl = dl.drop('DSL D-95')
  dl = pd.DataFrame(dl)
  dl = dl.reset_index()
  dl = dl['index']
  #dl = dl.drop('DSL D-95')
  #print(dl)
  train.drop(dl,axis=1,inplace=True)
  test.drop(dl,axis=1,inplace=True)
  return train,test


def preprocessing2(df):
  train = df[(df['year'] == 2015) |(df['year'] == 2016) | (df['year'] == 2017) | (df['year'] == 2018)]
  test = df[(df['year'] == 2019) |(df['year'] == 2020) | (df['year'] == 2021)]
  return train, test


def preprocessing3(train, test):
  train_x = train.drop(['DSL D-95'],axis=1)
  train_y = train['DSL D-95']
  test_x = test.drop(['DSL D-95'],axis=1)
  test_y = test['DSL D-95']
  #train_x.drop(['Unnamed: 0'],axis=1,inplace=True)
  #test_x.drop(['Unnamed: 0'],axis=1,inplace=True)
  return train_x, train_y, test_x, test_y


def filtering_7H(df):
  df = df[df['hour'] == 7]
  df = df.reset_index(drop = True)
  return df

def feature_selection(df):
  df['A+B'] = df['TI21022A(Catalyst T 1)'] + df['TI21022B(Catalyst T 2)']
  df = df[['PIC23106(Top P)', 'FIC23105(P/A RT Flow)', 'A+B','TI23118(D/O Liquid T)', 'TI23502(D/O Vapor T)']]
  return df


def log(df):
  col_list = df.columns
  for i in col_list:
    df.loc[:,i + 'log'] = np.log1p(df.loc[:,i])
  return df

def kalman_filter(df):
  col_list = df.columns
  
  for i in tqdm(col_list):
      current=0
      sum_c=[]
      z = df.loc[:, i]
      a = []           #필터링 된 피쳐(after)
      b = []           #필터링 전 피쳐(before)
      my_filter = KalmanFilter(dim_x=2,dim_z=1) #create kalman filter
      my_filter.x = np.array([[2.],[0.]])       # initial state (location and velocity)
      my_filter.F = np.array([[1.,1.], [0.,1.]])    # state transition matrix
      my_filter.H = np.array([[1.,0.]])    # Measurement function
      my_filter.P *= 1000.                 # covariance matrix
      my_filter.R = 5                      # state uncertainty
      my_filter.Q = Q_discrete_white_noise(dim = 2,dt=.1,var=.1) # process uncertainty   
      for k in z.values:
          my_filter.predict()
          my_filter.update(k)
          # do something with the output
          x = my_filter.x
          a.extend(x[0])
          b.append(k)
      sum_c=sum_c+a
      df.loc[:,'kf_X_'+str(i)]=sum_c
  return df

def mean_median_30(df):
  raw_cols_tr = df.columns

  mean_arr = []
  median_arr = []


  for column in raw_cols_tr:
      column_list = df[column].to_list()
      for i in range(30):
        mean_arr.append(column_list[i])
        median_arr.append(column_list[i])
          
      for i in range(30, len(column_list)):
          mean_arr.append(float(np.mean(column_list[i-30:i])))
          median_arr.append(float(np.median(column_list[i-30:i])))
      df[f'{column}_mean_30'] = mean_arr
      df[f'{column}_median_30'] = median_arr
      mean_arr = []
      median_arr = []
  return df



