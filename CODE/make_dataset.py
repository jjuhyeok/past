from glob import glob
from preprocessing import *

def make_dataset(input_path):

    pd.options.display.float_format = '{: .20f}'.format

    data = preprocessing1(input_path)
    
    data = Tree_sigma(data)

    train, test = preprocessing2(data)

    train_x, train_y, test_x, test_y = preprocessing3(train,test)
    # '+'상관관계 애들
    #train_x = train_x[['TIC23115(Feed1 T)','TI23029(Feed2 T)','TI23028(P/A RT T)','TI23120(F Zone T)','TI23121(SS T)','TI23502(D/O Vapor T)','TI23122(BTM T)','TI23119(OV T)','TI23118(D/O Liquid T)','FIC23133(R/D Flow)']]
    #test_x = test_x[['TIC23115(Feed1 T)','TI23029(Feed2 T)','TI23028(P/A RT T)','TI23120(F Zone T)','TI23121(SS T)','TI23502(D/O Vapor T)','TI23122(BTM T)','TI23119(OV T)','TI23118(D/O Liquid T)','FIC23133(R/D Flow)']]

    # EDA 기반
    #train_x = train_x[['FIC21192(F2 Flow)','TI23502(D/O Vapor T)','FIC23010(D/O Flow)','FIC25103(OVHD Flow 1)','FIC23133(R/D Flow)','FIC23110(OVHD Flow 2)']]
    #test_x = test_x[['FIC21192(F2 Flow)','TI23502(D/O Vapor T)','FIC23010(D/O Flow)','FIC25103(OVHD Flow 1)','FIC23133(R/D Flow)','FIC23110(OVHD Flow 2)']]

    
    #drop_list of corr
    #corr 상위 몇개만 사용할 지
    #14->15 될때 mae 급 상승(9->11)
    #18->19 될때 mae 급 감소(11->9)
    n = 10
    #train_x, test_x = drop_List(data,train_x,test_x, n)

    #train = filtering_7H(train)
    test = filtering_7H(test)
    
    #train_x, train_y, test_x, test_y = preprocessing3(train,test)



    #train_x = feature_selection(train_x)
    #test_x = feature_selection(test_x)

    #sns.boxplot(train_x)
    #sns.boxplot(test_x)


    ####Feature Engineering####

    #train_x = log(train_x)
    #test_x = log(test_x)
    
    #train_x = kalman_filter(train_x)
    #test_x = kalman_filter(test_x)
    
    #train_x = mean_median_30(train_x)
    #test_x = mean_median_30(test_x)

    print("==================================")
    print("==================================")
    print("=============complete=============")
    print("==================================")
    print("==================================")
    return train_x, train_y, test_x, test_y
