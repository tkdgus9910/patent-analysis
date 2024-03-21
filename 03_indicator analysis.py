
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
        
    #%% 1. HHI index 계산
    
    import indicator
    
    # 분야 HHI 변화 확인 : 2000년 이전
    data_input = data_preprocessed.loc[data_preprocessed['year_application'] < 2000, :].reset_index(drop = 1)
    print(indicator.calculate_HHI(data_input, 'applicant_rep_icon', 'id_publication'))
    
    # 분야 HHI 변화 확인 : 현재까지 
    data_input = data_preprocessed.loc[data_preprocessed['year_application'] < 2023, :].reset_index(drop = 1)
    print(indicator.calculate_HHI(data_input, 'applicant_rep_icon', 'id_publication'))
    
    
    #%%