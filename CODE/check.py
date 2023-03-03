import pandas as pd

df = pd.read_csv('\\Users\\ineeji\\Desktop\\past\\datas\\데이터합본_파생변수 제거.csv')
df['year'] = df['Unnamed: 0'].apply(lambda x : x.split()[0].split('-')[0])
df['hour'] = df['Unnamed: 0'].apply(lambda x : x.split()[1].split(' ')[0].split(':')[0])
df['hour'] = df['hour'].astype('int')
df['year'] = df['year'].astype('int')

test = df[(df['year'] == 2019) |(df['year'] == 2020) | (df['year'] == 2021)]

test = test[test['hour'] == 7]
test = test.reset_index(drop = True)

print("원래 test : %d"%len(test))



test1 = test.copy()
test1 = test1[test1['TI21022A(Catalyst T 1)'] > 360]


print("TI21022A(Catalyst T 1) 360 이상 test : %d,   target min : %lf"%(len(test1), test1['DSL D-95'].min())
)



test2 = test.copy()
test2 = test2[test2['DSL D-95'] > 360]


print("DSL D-95 360 이상 test : %d,   target min : %lf"%(len(test2), test2['DSL D-95'].min())
)





def drop_abnormal_region(df, target_col, std_threshold, rate_threshold):
    """
    데이터프레임에서 타겟 컬럼이 갑자기 떨어지는 구간을 찾아 해당 구간을 드랍시킵니다.
    """
    # 감소 구간 탐색을 위한 변수 초기화
    is_dropping = False
    start_idx = None
    
    # 드랍시킬 구간을 기록할 리스트 초기화
    drop_regions = []
    
    # 타겟 컬럼의 표준편차와 변화량에 대한 기준값 계산
    std = df[target_col].std()
    threshold = std * std_threshold
    #print(std,threshold)
    #print(df.loc[330:340.:])
    # 전체 데이터에 대해 반복
    for i in range(5, len(df)):
        # 이전 값과의 차이를 계산
        diff = abs(df[target_col][i] - df[target_col][i-5])
        
        # 감소 구간 탐색
        if diff / std >= rate_threshold:
            start_idx = i - 5
            #print("========================================")
            #print(df[target_col][i],df[target_col][i-5], i, i - 5)
            #print(diff, std, rate_threshold, start_idx,i-1)
            #print("========================================")
            for j in range(start_idx,i):
                drop_regions.append(j)
            #drop_regions.append((start_idx, i - 1))
            #is_dropping = True
        #elif is_dropping and diff / std > rate_threshold:
            #print(start_idx,i-1)
            #drop_regions.append((start_idx, i - 1))
            #is_dropping = False
    #print(drop_regions)
    # 드랍시킬 구간을 데이터프레임에서 제거
    print(drop_regions)
    '''
    for start, end in drop_regions:
        df = df.drop(df.index[start:end+1])'''
    df = df.drop(drop_regions, axis=0)
    #print(df.loc[5:15.:])
    return df


test3 = test.copy()
test3 = test3.reset_index(drop = True)
test3 = drop_abnormal_region(test3, 'DSL D-95', 1, 0.5)
test3 = test3.reset_index(drop = True)


print("Moving filter test : %d, target min %lf"%(len(test3), test3['DSL D-95'].min())
)


print(test3[test3['DSL D-95'] < 370])
