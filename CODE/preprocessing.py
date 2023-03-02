import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
import seaborn as sns
#from filterpy.kalman import KalmanFilter
#from filterpy.common import Q_discrete_white_noise
from tqdm import tqdm
from sklearn.decomposition import PCA


def preprocessing1(input_path):
  data = pd.read_csv('/Users/jeonjuhyeog/Documents/past/datas/데이터합본_파생변수 제거.csv')

  data['year'] = data['Unnamed: 0'].apply(lambda x : x.split()[0].split('-')[0])
  data['month'] = data['Unnamed: 0'].apply(lambda x : x.split()[0].split('-')[1])
  data['date'] = data['Unnamed: 0'].apply(lambda x : x.split()[0].split('-')[2])
  data['hour'] = data['Unnamed: 0'].apply(lambda x : x.split()[1].split(' ')[0].split(':')[0])

  data['year'] = data['year'].astype('int')
  data['month'] = data['month'].astype('int')
  data['date'] = data['date'].astype('int')
  data['hour'] = data['hour'].astype('int')
  return data


def drop_under_TI_360(data):
   data = data[data['TI21022A(Catalyst T 1)'] > 370]
   return data

def Tree_sigma(df):
  lower_out = df['DSL D-95'].mean() - df['DSL D-95'].std()*3
  #upper_out = df['DSL D-95'].mean() + df['DSL D-95'].std()*3
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


def plus_PCA(train_X, test_X, threshold=0.8, n_components=1):
    """
    feature들 간의 상관관계가 높은 feature들을 뽑아서 PCA를 시도한 후, 결과로 나온 피처를 추가하는 함수
    
    Args:
    train_X (pd.DataFrame): 학습 데이터셋 feature
    test_X (pd.DataFrame): 테스트 데이터셋 feature
    threshold (float): 상관계수 threshold (default: 0.8)
    n_components (int): PCA 컴포넌트 수 (default: 1)
    
    Returns:
    train_X_new (pd.DataFrame): 학습 데이터셋에 PCA 피처가 추가된 데이터프레임
    test_X_new (pd.DataFrame): 테스트 데이터셋에 PCA 피처가 추가된 데이터프레임
    """
    # 상관계수 행렬 계산
    corr_matrix = train_X.corr().abs()
    
    # threshold 이상의 상관계수를 가지는 feature들 추출
    high_corr_features = set()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if corr_matrix.iloc[i, j] >= threshold:
                high_corr_features.add(corr_matrix.columns[i])
                high_corr_features.add(corr_matrix.columns[j])
    high_corr_features = list(high_corr_features)
    
    # 추출된 feature들을 이용해 PCA를 적용하여 새로운 feature 생성
    pca = PCA(n_components=n_components)
    pca.fit(train_X[high_corr_features])
    pca_features_tr = pca.transform(train_X[high_corr_features])
    pca_features_te = pca.transform(test_X[high_corr_features])
    
    # 새로운 feature 추가
    train_X_new = train_X.copy()
    test_X_new = test_X.copy()
    for i in range(n_components):
        train_X_new[f"PCA_{i+1}"] = pca_features_tr[:, i]
        test_X_new[f"PCA_{i+1}"] = pca_features_te[:, i]
    train_X_new.drop(high_corr_features, axis = 1, inplace=True)
    test_X_new.drop(high_corr_features, axis = 1, inplace=True)
    return train_X_new, test_X_new



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



