from os.path import join
import pandas as pd
from make_dataset import make_dataset
from sklearn.metrics import mean_absolute_error

path = '\\Users\\ineeji\\Desktop\\새 폴더\\Ineeji\\datas\\데이터합본_파생변수 제거.csv'
train_x, train_y, test_x, test_y = make_dataset(path)

