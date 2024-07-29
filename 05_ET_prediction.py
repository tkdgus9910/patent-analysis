# -*- coding: utf-8 -*-
"""
Created on Fri May 31 15:14:03 2024

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
    directory = directory.replace("\\", "/") # window
    sys.path.append(directory+'/submodule')
    import preprocess
            
    # 위즈도메인 csv 파일이 존재하는 디렉토리 설정
    directory = os.environ['directory_path']
    directory += '우주/'
    
    file_list = os.listdir(directory)
    
    # 데이터 로드 
    with open(directory + 'data_preprocessed.pkl', 'rb') as file:  
        data_preprocessed = pickle.load(file)  