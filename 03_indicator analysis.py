
if __name__ == '__main__':
    
    import os
    import sys
    import pandas as pd
    import numpy as np     
    import warnings
    import pickle 
    
    warnings.filterwarnings("ignore")
    
    # 서브모듈 경로 설정
    directory = os.path.dirname(os.path.abspath(__file__))
        
    # 위즈도메인 csv 파일이 존재하는 디렉토리 설정
    directory_ = 'D:/data/우주/'
    
    directory = directory.replace("\\", "/") # window
    sys.path.append(directory+'/submodule')
    import preprocess
    
    # 데이터 로드 
    with open(directory_ + 'data_preprocessed.pkl', 'rb') as file:  
        data_preprocessed = pickle.load(file)  
        
    #%% 1. 분야 경쟁 지수 : HHI-index 계산 
    
    # United States Department of Justice 기준
    # 1500 미만 : competitive marketplace
    # 1500~2500 : moderately concentrated
    # 2500 이상 : highly concentrated
    
    ###
    import indicator
    
    # 분야 HHI 변화 확인 : 2000년 이전
    data_input = data_preprocessed.loc[data_preprocessed['year_application'] < 2000, :].reset_index(drop = 1)
    print(indicator.calculate_HHI(data_input, 'applicant_rep_icon', 'id_publication'))
    
    # 분야 HHI 변화 확인 : 현재까지 
    data_input = data_preprocessed.loc[data_preprocessed['year_application'] < 2023, :].reset_index(drop = 1)
    print(indicator.calculate_HHI(data_input, 'applicant_rep_icon', 'id_publication'))
    
    #%% 2. 연도 별 피인용수 보정
    
    # 보정 필요 확인
    import matplotlib.pyplot as plt 
    data_input = data_preprocessed.loc[data_preprocessed['year_application'] >= 1974, :].reset_index(drop = 1)
    grouped = data_input.groupby('year_application')
    mean_std = grouped['citation_forward_domestic_count'].agg(['mean', 'std'])
    
    plt.plot(mean_std['mean'], 'bs')
    xticks = mean_std.index
    xticks = [i for i in xticks if i%4 ==0 ]
    plt.xticks(xticks)
    plt.grid(True) 
    plt.axvline(x = 2016, color = 'r', label = '18-month publication')
    plt.axvline(x = 2006, color = 'r', label = '18-month publication')

    # 2006~2016년 데이터의 경우 보정
    data_input = data_preprocessed.loc[data_preprocessed['year_application'] <= 2005, :].reset_index(drop = 1)
    mean = data_input['citation_forward_domestic_count'].mean()
    std = data_input['citation_forward_domestic_count'].std()
    
    # 방안 1. 간극 보정(로그함수)
    data_preprocessed['citation_forward_domestic_count_sqrt'] = data_input['citation_forward_domestic_count'].apply(lambda x : np.sqrt(x))
    
    from scipy.stats import zscore
    
    # 방안 2. z-score
    grouped = data_input.groupby('year_application')
    mean_std = grouped['citation_forward_domestic_count'].agg(['mean', 'std'])
    merged = pd.merge(data_input, mean_std, left_on= 'year_application', right_index= True)
    
    data_input['citation_forward_domestic_count_z'] = (merged['citation_forward_domestic_count'] - merged['mean']) / merged['std']
    data_input['citation_forward_domestic_count_z'] = data_input['citation_forward_domestic_count_z'].fillna(0)
    
    
    #%% 3. 주요 출원인 추출
    import indicator
    
    # 기준 1. 특허수
    grouped = data_input.groupby('applicant_rep_icon')
    result = grouped['citation_forward_domestic_count'].agg(['count'])
    result = result.reset_index()
    result = result.sort_values(by = 'count', ascending = 0).reset_index(drop = 1)
    
    # 기준 2-1. 피인용수 단순합
    grouped = data_input.groupby('applicant_rep_icon')
    result = grouped['citation_forward_domestic_count'].agg(['sum'])
    result = result.reset_index()
    result = result.sort_values(by = 'sum', ascending = 0).reset_index(drop = 1)
    
    # 기준 2-2. 로그보정 합 -> best?
    grouped = data_input.groupby('applicant_rep_icon')
    result = grouped['citation_forward_domestic_count_sqrt'].agg(['sum'])
    result = result.reset_index()
    result = result.sort_values(by = 'sum', ascending = 0).reset_index(drop = 1)
    
    # 기준 3. 기술력 지수
    result = indicator.calculate_TS(data_input, 'applicant_rep_icon', 'citation_forward_domestic_count')
    
    # 기준 4. 피인용수_연도 보정 - 수정필요
    grouped = data_input.groupby('applicant_rep_icon')
    result = grouped['citation_forward_domestic_count_z'].agg(['sum'])
    result = result.reset_index()
    result = result.sort_values(by = 'sum', ascending = 0).reset_index(drop = 1)
    
