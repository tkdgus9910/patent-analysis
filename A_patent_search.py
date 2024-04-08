
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
    directory_ = 'D:/OneDrive/연구실 과제(현대차_진섭)/2. 기능분석/특허 데이터/Raw/'
    
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
    

    #%% data filtering- 1. F-term, IPC 장전
    data_input = data_.loc[data_['year_application'] >= 2000 , :].reset_index(drop = 1) # 1000건 제외
    
    
    #%% 2. 대표청구항 분해
    import re 
    
    # 2-A. rule-based splitting
    # data_input['claims_decomposed'] = data_input['claims_rep'].apply(lambda x: re.split(r';|:', x))
    # data_input['claims_decomposed'] = data_input['claims_decomposed'].apply(lambda x: [i.strip() for i in x if len(i) >= 30])
    
    #%% 반복수행
    from openai import OpenAI
    client = OpenAI()
    
    # data_input['claims_decomposed'] = [[] for i in data_input['claims']]
    
    for idx, row in data_input.iterrows() : 
        if len(data_input['claims_decomposed'][idx]) == 0 : 
                
            print(idx)
            claim = row['claims_rep']
            prompt = """Decomposed and simplified the complex sentence into clearer, more manageable parts
Q : 1. A connection for an open clamp, comprising clamping band means including clamping band portions adapted to overlap in the installed position, tightening means for tightening the clamp about an object to be fastened thereby, connecting means for connecting overlapping band portions including at least one hook-shaped means performing a guide function during tightening of the clamp and extending outwardly from the inner band portion, said hook-shaped means being operable to engage in at least one aperture in the outer band portion, and abutment surface means in said hook-shaped means for preventing disengagement of the connecting means during the entire tightening operation.
A : ['A connection for an open clamp is described.',
 'This connection includes a clamping band with portions designed to overlap when installed.',
 'There are means for tightening the clamp around an object that needs to be fastened.',
 'For connecting the overlapping band portions, it features at least one hook-shaped element.',
 'This hook-shaped element aids in guiding during the tightening process and protrudes outward from the inner band portion.',
 'It is designed to latch into at least one aperture on the outer band portion.',
 'Additionally, the hook-shaped element has an abutment surface.',
 'This surface prevents the disengagement of the connecting elements throughout the tightening process.']   
Q : """
    
            prompt += claim #원하는 문서
            prompt += "\nA : "
            
            instruction = "Just return a result as a Python list format"
            
            completion = client.chat.completions.create(
                # model="gpt-4-0125-preview",
                model="gpt-3.5-turbo-0125",
                messages=[
                {"role": "user", 
                 "content": prompt},{"role" : "system", "content" : instruction}
                ])
            
            data_input['claims_decomposed'][idx] = completion.choices[0].message.content
    
    #%%
    
    import pickle
    
    with open(directory_+ 'claims_decomposed.pkl', 'wb') as f:
        pickle.dump(data_input, f)
    
    # with open(directory_ + 'cl')
    
    
    
    
    
            
    
    