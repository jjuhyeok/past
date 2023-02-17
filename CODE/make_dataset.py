from glob import glob
from preprocessing import *

def make_dataset(input_path):

    pd.options.display.float_format = '{: .20f}'.format
    data = preprocessing1(input_path)

    data = Tree_sigma(data)

    #drop_list of corr
    n = 5 #corr 상위 몇개만 사용할 지
    dl = drop_List(data, n)

    train, test = preprocessing2(data)
    
    train_x, train_y, test_x, test_y = preprocessing3(train,test)
    train_x = filtering_7H(train_x)
    test_x = filtering_7H(test_x)

    train_x = feature_selection(train_x)
    test_x = feature_selection(test_x)

    print("==================================")
    print("==================================")
    print("=============complete=============")
    print("==================================")
    print("==================================")
    return train_x, train_y, test_x, test_y
