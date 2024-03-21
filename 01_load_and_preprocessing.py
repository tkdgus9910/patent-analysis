# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 17:28:37 2024

@author: tmlab
"""


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
    
    file_list = os.listdir(directory_)
    data = pd.DataFrame()
    
    for file in file_list : 
        if '.csv' in file : 
            temp_data = pd.read_csv(directory_ + file, skiprows = 4)
            temp_data['file'] = file
            data = pd.concat([data, temp_data], axis = 0).reset_index(drop =1)
               
    data_ = preprocess.wisdomain_prep(data)    
    
    
    # 전처리 후 경로에 저장 
    with open(directory_ + 'data_preprocessed.pkl', 'wb') as file:  
        pickle.dump(data_, file)  