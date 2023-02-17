import pandas as pd
import numpy as np
import glob

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

def drop_List(df, n):
  dl = df.corr()['DSL D-95'].map(abs).sort_values(ascending = False)[n:]
  dl = pd.DataFrame(dl)
  dl = dl.reset_index()
  dl = dl['index']
  return dl


def preprocessing2(df):
  train = df[(df['year'] == 2015) |(df['year'] == 2016) | (df['year'] == 2017) | (df['year'] == 2018)]
  test = df[(df['year'] == 2019) |(df['year'] == 2020) | (df['year'] == 2021)]
  return train, test


def preprocessing3(train, test):
  train_x = train.drop(['DSL D-95'],axis=1)
  train_y = train['DSL D-95']
  test_x = test.drop(['DSL D-95'],axis=1)
  test_y = test['DSL D-95']
  train_x.drop(['Unnamed: 0'],axis=1,inplace=True)
  test_x.drop(['Unnamed: 0'],axis=1,inplace=True)
  return train_x, train_y, test_x, test_y


def filtering_7H(df):
  df = df[df['hour'] == 7]
  df = df.reset_index(drop = True)
  return df

def feature_selection(df):
  df['A+B'] = df['TI21022A(Catalyst T 1)'] + df['TI21022B(Catalyst T 2)']
  df = df[['PIC23106(Top P)', 'FIC23105(P/A RT Flow)', 'A+B','TI23118(D/O Liquid T)', 'TI23502(D/O Vapor T)']]
  return df
