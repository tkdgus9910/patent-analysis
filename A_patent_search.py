
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
    
    # data_input = data_.sample(500).reset_index(drop = 1)
    data_input = data_.loc[data_['year_application'] >= 2000 , :].reset_index(drop = 1) # 1000건 제외
    
    #%% embedding definition
    
    directory_ = 'D:/OneDrive/연구실 과제(현대차_진섭)/2. 기능분석/2차워크숍 기초자료/'
    
    definitions = pd.read_excel(directory_ + '분리결합체계_정의.xlsx',
                                sheet_name = 'Mechanical Attach_Detach_2nd')
    definitions['definition'] = definitions['definition'].apply(lambda x : x.strip())
    # definition_2nd = {}
    definition_2nd = dict(zip(definitions['category'], definitions['definition']))
    definition_2nd = dict(sorted(definition_2nd.items()))
    
    
    #%% 2. embedding text
    from sentence_transformers import SentenceTransformer
    
    embedding_model = SentenceTransformer('AI-Growth-Lab/PatentSBERTa')
    # embeddings_title = embedding_model.encode(data_input['title'] ,device='cuda')
    embeddings_TA = embedding_model.encode(data_input['TA'] ,device='cuda')
    embeddings_claims_rep = embedding_model.encode(data_input['claims_rep'] ,device='cuda')
    #%%
    
    embeddings_definitions = np.array([]).reshape(-1,768)
    
    for category, definition in definition_2nd.items() :
        
        definition = category + '. ' + definition
        print(definition)
        temp = embedding_model.encode(definition)
        embeddings_definitions = np.vstack((embeddings_definitions, temp))
        
    
    #%%
    
    from sklearn.metrics.pairwise import cosine_similarity

    distance_matrix = cosine_similarity(embeddings_claims_rep,embeddings_definitions) #p2p
    
    
    top10_indices_by_columns = np.argsort(distance_matrix, axis = 0)[-20:] #top 10 indices
    top10_indices_by_columns = top10_indices_by_columns[::-1] # reverse
    
    result = pd.DataFrame()
    
    for col_idx in range(0, top10_indices_by_columns.shape[1]) :
        print(col_idx)
        rank = 1
        for row_idx in top10_indices_by_columns[:,col_idx] :
            print(row_idx)
            
            # 정의 및 목차 관련
            category = list(definition_2nd.keys())[col_idx]
            definition = list(definition_2nd.values())[col_idx]
            
            
            # 특허 관련
            similarity = distance_matrix[row_idx, col_idx]
            id_wisdomain = data_input['id_wisdomain'][row_idx]
            claims_rep = data_input['claims_rep'][row_idx]
        
        
            data_temp = pd.DataFrame([{'category' : category,
                                      'definition' : definition,
                                      'rank' : rank,
                                      'similarity' : similarity,
                                      'id_wisdomain' : id_wisdomain,
                                      'claims_rep' : claims_rep}])
            result = pd.concat([result, data_temp], axis= 0 )
            rank += 1
            
    
    result = result.reset_index(drop = 1)
    result.to_excel(directory_ + 'match_patent_pilot.xlsx')
    
    #%% 3. 