# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 15:06:03 2024

@author: tmlab
"""

if __name__ == '__main__':
    
    import os
    import sys
    import pandas as pd
    import numpy as np     
    import warnings
    import pickle 
    import ast
    import re    
    from scipy import stats
    from sentence_transformers import SentenceTransformer
    from umap import UMAP
    
    from copy import copy 
    from sklearn.feature_extraction.text import CountVectorizer
    from hdbscan import HDBSCAN
    from bertopic.representation import MaximalMarginalRelevance
    from bertopic import BERTopic
    
    from bertopic.dimensionality import BaseDimensionalityReduction
    from sklearn.feature_extraction import text 
    from sklearn.feature_extraction import _stop_words
    from copy import copy
    from scipy import stats
    
    from sklearn.metrics.pairwise import cosine_similarity
    
    def bertopic_wrapper(min_cluister_size, docs, embeddings, cs_method,return_) :
        
        # Fit BERTopic without actually performing any dimensionality reduction
                
        cluster_model = HDBSCAN(min_cluster_size= min_cluister_size, 
                                # min_samples = int(min_cluister_size/5),                            
                                min_samples = 1,                            
                                cluster_selection_method= cs_method, 
                                # cluster_selection_epsilon='leaf', 
                                prediction_data=True,
                                gen_min_span_tree=1)
        
        cluster_model.fit(embeddings)

        if return_ == 'just_performance' :
            
            return(cluster_model)
        
        else : 
            
            topic_model = BERTopic(
              # Pipeline models
              embedding_model=embedding_model,
              umap_model=empty_dimensionality_model,
              hdbscan_model=cluster_model,
              vectorizer_model=vectorizer_model,
              representation_model=representation_model,
              calculate_probabilities=True,
              # Hyperparameters
              top_n_words=10,
              # verbose=True,
              nr_topics = 30,
            )
            
            topic_model.fit(docs, embeddings)    
            
            # Train model
            return(topic_model)
    #%%
    warnings.filterwarnings("ignore")
    
    # 서브모듈 경로 설정
    directory = os.path.dirname(os.path.abspath(__file__))
        
    # 위즈도메인 csv 파일이 존재하는 디렉토리 설정
    directory_ = 'D:/OneDrive/연구실 과제(현대차_진섭)/2. 기능분석/특허 데이터/Raw/'
    
    directory = directory.replace("\\", "/") # window
    sys.path.append(directory+'/submodule')
    import preprocess
    
    directory_ = 'D:/OneDrive/연구실 과제(현대차_진섭)/2. 기능분석/2차워크숍 기초자료/'
    
    with open(directory_+ 'claims_decomposed.pkl', 'rb') as f:
        data_input = pickle.load(f)
        
    #%%
    data_input = data_input.loc[data_input['claims_decomposed'].apply(lambda x : len(x) > 0 ),:].reset_index(drop = 1)
    #%%
    
    data_input['claims_decomposed_'] = data_input['claims_decomposed'].apply(lambda x : re.sub("'s", "`s", x))
    data_input['claims_decomposed_'] = data_input['claims_decomposed_'].apply(lambda x : re.sub("n't", "n`t", x))
    # data_input['claims_decomposed'] = data_input['claims_decomposed'].apply(lambda x : ast.literal_eval(x))
    
    exception = []
    for idx, row in data_input.iterrows() : 
        temp = row['claims_decomposed_']
        try : 
            temp = ast.literal_eval(temp)
            data_input['claims_decomposed_'][idx] = temp
        except :
            data_input['claims_decomposed_'][idx] = np.nan
            exception.append(temp)
            
    # embedding definitions
    
    definitions = pd.read_excel(directory_ + '분리결합체계_정의.xlsx', #category, definition으로 이루어진 검색어
                                sheet_name = '1st')
    
    definitions['definition'] = definitions['definition'].apply(lambda x : x.strip())
    definition_mechanical = definitions['category'][0] + '. ' + definitions['definition'][0]
    definition_electrical = definitions['category'][1] + '. ' + definitions['definition'][1]
    
    definitions = pd.read_excel(directory_ + '분리결합체계_정의.xlsx', #category, definition으로 이루어진 검색어
                                # sheet_name = 'Electrical Attach_Detach_2nd')
                                sheet_name = 'Mechanical Attach_Detach_2nd')
    
    definitions['definition'] = definitions['definition'].apply(lambda x : x.strip())
    definition_2nd = dict(zip(definitions['category'], definitions['definition']))
    definition_2nd = dict(sorted(definition_2nd.items()))
    
    
    
    # 정의 embedding
    embedding_model = SentenceTransformer('AI-Growth-Lab/PatentSBERTa')
    embeddings_definitions = np.array([]).reshape(-1,768)
    embedding_mechanical = embedding_model.encode(definition_mechanical)
    embedding_electrical = embedding_model.encode(definition_electrical)
    
    # 1. 대표청구항과 정의 비교 embedding text
    for category, definition in definition_2nd.items() :
        
        definition = category + '. ' + definition
        
        print(definition)
        temp = embedding_model.encode(definition)
        
        embeddings_definitions = np.vstack((embeddings_definitions, temp))
    
    #%%
    
    data_input = data_input.loc[data_input['claims_decomposed_'].apply(lambda x : str(x) != 'nan' ),:].reset_index(drop = 1)
    data_input['claims_decomposed_10'] = [[] for i in range(len(data_input['claims_decomposed']))]
    
    for idx, row in data_input.iterrows() : 
        
        for text in row['claims_decomposed_'] :    
            if type(text) == list :
                text = text[0]
            
            if len(text.split()) >= 10 :
                data_input['claims_decomposed_10'][idx].append(text)
        
        
    # data_input['claims_decomposed_len'] = data_input['claims_decomposed_len'].apply(lambda x : [len(i) for i in x])
    # data_input['claims_decomposed_'] = data_input['claims_decomposed_'].apply(lambda x : [i for i in x if len(i.split()) >= 10])
    # data_input = data_input.loc[data_input['claims_decomposed_'].apply(lambda x : str(x) != 'nan' ),:].reset_index(drop = 1)
    # 진짜 오래걸림
    data_input['claims_decomposed_embeddings'] = data_input['claims_decomposed_10'].apply(lambda x : embedding_model.encode(x ,device='cuda'))
    
    #%%
    
    
    temp = copy(data_input)
    temp = temp.dropna(subset = ['claims_decomposed_10'])
    
    
    #%%
    
    embeddings_total = np.array(data_input['claims_decomposed_embeddings'])
    embeddings_total = [item for sublist in embeddings_total for item in sublist]
    embeddings_total = np.array(embeddings_total)
    
    similarity_matrix = cosine_similarity(embeddings_total, embeddings_definitions)

    threshold = np.median(similarity_matrix) + 2*np.std(similarity_matrix)
    print(threshold)
    
    
    #%% iter 1. 기준 이상의 문장들 필터링
    
    data_related = pd.DataFrame()
    definition_idx = 1 # 정의 설정
    # print()
    for idx, row in data_input.iterrows() : 
        
        print(list(definition_2nd.keys())[definition_idx] + '_'+str(idx))
        embeddings_temp = row['claims_decomposed_embeddings']
        if len(embeddings_temp) == 0 :continue
        
        
        similarity_matrix = cosine_similarity(embeddings_temp, embeddings_definitions)
        indices = np.where(similarity_matrix > threshold,1,0)
        indices = list(np.where(indices[:,definition_idx] == 1)[0]) # definition_idx번째 관련 특허들
        if len(indices) == 0 : continue
    
        # sentences = [row['claims_decomposed'][i] for i in indices]
        
        for idx_ in indices : 
            
            sentence = row['claims_decomposed_10'][idx_]
            similarity_2nd = similarity_matrix[idx_][definition_idx]
            similarity_mechanical = cosine_similarity([embeddings_temp[idx_]], [embedding_mechanical])[0][0]
            similarity_electrical = cosine_similarity([embeddings_temp[idx_]], [embedding_electrical])[0][0]
    
            data_temp = pd.DataFrame([{'id_wisdomain' : row['id_wisdomain'],
                                      'claims_rep' : row['claims_rep'],
                                      'claims_decomposed' : sentence, 
                                      'similarity_2nd' : similarity_2nd,
                                      'similarity_mechanical' : similarity_mechanical,
                                      'similarity_electrical' : similarity_electrical}])
            
            data_related = pd.concat([data_related, data_temp], axis= 0)
    
    data_related['similarity_mechanical_z'] = stats.zscore(data_related['similarity_mechanical'], nan_policy='omit', ddof=0)
    data_related['similarity_electrical_z'] = stats.zscore(data_related['similarity_electrical'], nan_policy='omit', ddof=0)
    data_related['similarity_indicator'] = data_related['similarity_mechanical_z'] - data_related['similarity_electrical_z']
    # data_related['similarity_indicator'] = data_related['similarity_electrical_z'] - data_related['similarity_mechanical_z']
    
    temp_data = data_related.loc[(data_related['similarity_mechanical_z'] >= 0), :].reset_index(drop = 1)
    # temp_data = data_related.loc[(data_related['similarity_electrical_z'] >= -0.5), :].reset_index(drop = 1)
    temp_data = temp_data.loc[(temp_data['similarity_indicator'] >= 0), :].reset_index(drop = 1)
    
    
    #%% step 2.  clustering
    
    docs = list(temp_data['claims_decomposed'])
    embeddings_claims_decomposed = embedding_model.encode(docs ,device='cuda')
    
    
    umap_model = UMAP(n_neighbors=15, n_components = 10, min_dist=0.0, metric='cosine', random_state=42)
    embeddings_claims_decomposed_50 = umap_model.fit_transform(embeddings_claims_decomposed) 
    
    empty_dimensionality_model = BaseDimensionalityReduction()
    
    custom_stopwords = ['described', 'apparatus', 'include' , 'includes', 'consists', 'features', 
                        'feature', 'device', 'allows', 'allow', 'use', 'method', 'comprises', 'various', 'section']
    stop_words = list(_stop_words.ENGLISH_STOP_WORDS.union(custom_stopwords))
    
    vectorizer_model = CountVectorizer(stop_words= stop_words,
                                       min_df = 0.0001,
                                       max_df = 0.6,
                                        ngram_range = (1,2),
                                       preprocessor=lambda x: re.sub(r'\d+', '', x).lower().strip()
                                       
                                       )
    
    representation_model = MaximalMarginalRelevance(diversity = 0.3) # default = 0.1
    

    embedding_model = SentenceTransformer('AI-Growth-Lab/PatentSBERTa')
    docs_input = docs
    embeddings_input = embeddings_claims_decomposed_50
    
    #%% step 3. 파라미터 튜닝 (optional)
    
    DBCVs = [] # Density-based clustering validation
    CPs = [] 
    outliers_counts = []
    cluster_N_counts = []
    max_cluster_sizes = []
    
    K = range(5, 101, 5)
    
    for size in K : 
        
        topic_model = bertopic_wrapper(size, 
                                       docs_input, 
                                       embeddings_input, 'leaf', return_ = 'just_performance')
        
        DBCVs.append(topic_model.relative_validity_)
        CPs.append(np.mean(topic_model.cluster_persistence_))
        
        cluster_N_counts.append(len(set(topic_model.labels_)))
        outliers_counts.append(np.count_nonzero(topic_model.labels_ == -1))
        max_cluster_sizes.append(np.count_nonzero(topic_model.labels_ == 0))
        
        print(size)
    
    import matplotlib.pyplot as plt
    
    # 시각화-DBCV
    plt.figure(figsize=(12,8))
    plt.subplot(211)
    plt.plot(K, DBCVs, 'bx-')
    # plt.xlabel('min_cluister_size')
    plt.ylabel('DBCVs', fontsize=15)
    
    # 시각화-DBCV    
    plt.subplot(212)
    plt.plot(K, CPs, 'rx-')
    plt.xlabel('Min_cluister_size', fontsize=15)
    plt.ylabel('Mean_cluster_persistences', fontsize=15)
    
    
    # 시각화-Outlier
    plt.figure(figsize=(12,8))
    plt.subplot(311)
    outliers_ratios = [round(i / len(docs_input), 3) for i in outliers_counts]
    plt.plot(K, outliers_ratios, 'rx-')
    
    # plt.xlabel('min_cluister_size', fontsize=15)
    plt.ylabel('outliers_ratio', fontsize=15)
    
    # 시각화
    plt.subplot(312)
    plt.plot(K, cluster_N_counts, 'gx-')
    # plt.xlabel('min_cluister_size')
    plt.ylabel('cluster_counts', fontsize=15)
    
    # 시각화
    plt.subplot(313)
    max_cluster_sizes = [round(i / len(docs_input), 3) for i in max_cluster_sizes]
    plt.plot(K, max_cluster_sizes, 'yx-')
    plt.xlabel('min_cluister_size', fontsize=15)
    plt.ylabel('biggest_cluster_ratio', fontsize=15)
    plt.show()
    
    max_idx = DBCVs.index(max(DBCVs))
    
    #%% step 4. 결과 확인
    
    topic_model1 = bertopic_wrapper(10, 
                                    docs_input, 
                                    embeddings_input,
                                    'leaf', 
                                    return_ = 'final_result')
    
    result_topic_info = topic_model1.get_topic_info()
    result_document_info = topic_model1.get_document_info(docs_input) # 40개의 도메인

    print(list(definition_2nd.keys())[definition_idx])
    print(list(definition_2nd.values())[definition_idx])
    topics_ = topic_model1.topics_
    probs_ = topic_model1.probabilities_
    
    new_topics = topic_model1.reduce_outliers(docs_input, topics_, 
                                              probabilities=probs_, 
                                              threshold=0.1, strategy="probabilities")

    topic_model1.update_topics(docs, topics=new_topics,
                               vectorizer_model=vectorizer_model,
                               representation_model=representation_model,)
    result_topic_info = topic_model1.get_topic_info()
    
    
    #%% step 5. 결과 저장
    directory_output = 'D:/data/현대차/'
    result_document_info = pd.concat([result_document_info, temp_data], axis = 1)
    file_name = list(definition_2nd.keys())[definition_idx]
    
    file_name += '_mechanical_docs'
    file_name += '.xlsx'
    result_document_info.to_excel(directory_output+ file_name)
    
    file_name = list(definition_2nd.keys())[definition_idx]
    file_name += '_mechanical_topics'
    file_name += '.xlsx'
    result_topic_info.to_excel(directory_output+ file_name)
    # val = list(definition_2nd.values())[definition_idx]
    
    
    #%% 반복 수행
    
    for definition_idx in range(len(definitions)) :
    # for definition_idx in [11] :
        
        data_related = pd.DataFrame()
        
        # print()
        for idx, row in data_input.iterrows() : 
            
            print(list(definition_2nd.keys())[definition_idx] + '_'+str(idx))
            embeddings_temp = row['claims_decomposed_embeddings']
            if len(embeddings_temp) == 0 :continue
            
            similarity_matrix = cosine_similarity(embeddings_temp, embeddings_definitions)
            indices = np.where(similarity_matrix > threshold,1,0)
            indices = list(np.where(indices[:,definition_idx] == 1)[0]) # definition_idx번째 관련 특허들
            if len(indices) == 0 : continue
        
            # sentences = [row['claims_decomposed'][i] for i in indices]
            
            for idx_ in indices : 
                
                sentence = row['claims_decomposed_10'][idx_]
                similarity_2nd = similarity_matrix[idx_][definition_idx]
                similarity_mechanical = cosine_similarity([embeddings_temp[idx_]], [embedding_mechanical])[0][0]
                similarity_electrical = cosine_similarity([embeddings_temp[idx_]], [embedding_electrical])[0][0]
        
                data_temp = pd.DataFrame([{'id_wisdomain' : row['id_wisdomain'],
                                          'claims_rep' : row['claims_rep'],
                                          'claims_decomposed' : sentence, 
                                          'similarity_2nd' : similarity_2nd,
                                          'similarity_mechanical' : similarity_mechanical,
                                          'similarity_electrical' : similarity_electrical}])
                
                data_related = pd.concat([data_related, data_temp], axis= 0)
        
        
        data_related['similarity_mechanical_z'] = stats.zscore(data_related['similarity_mechanical'], nan_policy='omit', ddof=0)
        data_related['similarity_electrical_z'] = stats.zscore(data_related['similarity_electrical'], nan_policy='omit', ddof=0)
        # data_related['similarity_indicator'] = data_related['similarity_mechanical_z'] - data_related['similarity_electrical_z']
        data_related['similarity_indicator'] = data_related['similarity_electrical_z'] - data_related['similarity_mechanical_z']
        
        # temp_data = data_related.loc[(data_related['similarity_mechanical_z'] > 0), :].reset_index(drop = 1)
        temp_data = data_related.loc[(data_related['similarity_electrical_z'] > 0), :].reset_index(drop = 1)
        temp_data = temp_data.loc[(temp_data['similarity_indicator'] > 0), :].reset_index(drop = 1)
    
        
        # clustering
        docs = list(temp_data['claims_decomposed'])
        embeddings_claims_decomposed = embedding_model.encode(docs ,device='cuda')
    
        umap_model = UMAP(n_neighbors=15, n_components = 10, min_dist=0.0, metric='cosine', random_state=42)
        embeddings_claims_decomposed_50 = umap_model.fit_transform(embeddings_claims_decomposed) 
        
        empty_dimensionality_model = BaseDimensionalityReduction()
        
        custom_stopwords = ['described', 'apparatus', 'include' , 'includes', 'consists', 'features', 
                            'feature', 'device', 'allows', 'allow', 'use', 'method', 'comprises', 'various', 'section', 'present']
        stop_words = list(_stop_words.ENGLISH_STOP_WORDS.union(custom_stopwords))
        
        vectorizer_model = CountVectorizer(stop_words= stop_words,
                                           min_df = 0.001,
                                           max_df = 0.5,
                                           ngram_range = (1,2),
                                           preprocessor=lambda x: re.sub(r'\d+', '', x).lower().strip()
                                           
                                           )
        
        representation_model = MaximalMarginalRelevance(diversity=0.3) # default = 0.1
            
        embedding_model = SentenceTransformer('AI-Growth-Lab/PatentSBERTa')
        docs_input = docs
        embeddings_input = embeddings_claims_decomposed_50
    
     
        topic_model1 = bertopic_wrapper(10, 
                                        docs_input, 
                                        embeddings_input,
                                        'leaf', 
                                        return_ = 'final_result')
        
        # 아웃라이어 할당
        # topics_ = topic_model1.topics_
        # new_topics = topic_model1.reduce_outliers(docs_input, topics_, strategy="c-tf-idf", threshold= 0.35)

        # topic_model1.update_topics(docs, topics=new_topics,
        #                            vectorizer_model=vectorizer_model,
        #                            representation_model=representation_model,)
        
        # result_topic_info = topic_model1.get_topic_info()
        
        result_topic_info = topic_model1.get_topic_info()
        result_topic_info = result_topic_info.loc[result_topic_info['Topic'] != -1, :].reset_index(drop = 1)
        result_document_info = topic_model1.get_document_info(docs_input) # 40개의 도메인
        
        print(list(definition_2nd.keys())[definition_idx])
        print(list(definition_2nd.values())[definition_idx])
        
        directory_output = 'D:/OneDrive/연구실 과제(현대차_진섭)/2. 기능분석/2차워크숍 기초자료/level3/'
        result_document_info = pd.concat([result_document_info, temp_data], axis = 1)
        file_name = list(definition_2nd.keys())[definition_idx]
        
        # file_name += '_mechanical_docs'
        file_name += '_electrical_docs'
        file_name += '.xlsx'
        result_document_info.to_excel(directory_output+ file_name)
        
        file_name = list(definition_2nd.keys())[definition_idx]
        # file_name += '_mechanical_topics'
        file_name += '_electrical_topics'
        file_name += '.xlsx'
        result_topic_info.to_excel(directory_output+ file_name)
            # val = list(definition_2nd.values())[definition_idx]
    
    #%% 반복 레이블링
    
    from openai import OpenAI
    client = OpenAI()
    file_list = os.listdir(directory_output)
    file_list = [i for i in file_list if '_topics' in i]
    
    for file in file_list :
        result_topic_info = pd.read_excel(directory_output+file, index_col=0)
        result_topic_info['label3'] = ''
        
        for idx, row in result_topic_info.iterrows() : 
            print(idx)
            keywords = str(row['Representation'])
            instruction= """Provide results simply without explanation."""
            prompt = """Based on the given topic modelling results, suggest sub-technology topic names related to "{}" technology respectively.
- Keywords are listed in descending order according to their importance.
Q: ['body', 'mounting', 'joint portion', 'components', 'mechanical connection', 'vehicle', 'treatment portion', 'sensor', 'includes mechanical', 'sheath portion']
A : Body-Mounted Couplings
Q :""".format(file.split('_')[0])
            prompt += keywords
            prompt += "\nA: "
            
            completion = client.chat.completions.create(
                model="gpt-4-0125-preview",
                # model="gpt-3.5-turbo-0125",
                messages=[
                {"role": "user", 
                 "content": prompt},{"role" : "system", "content" : instruction}
                ])
            
            result_topic_info['label3'][idx] = completion.choices[0].message.content
        
        
        result_topic_info.to_excel(directory_output+file )
        
    #%% 도메인 할당
    
    # 도메인 로드
    
    #%% [backup] 대표 청구항 할당
    
    # embeddings_title = embedding_model.encode(data_input['title'] ,device='cuda')
    embeddings_TA = embedding_model.encode(data_input['TA'] ,device='cuda')
    embeddings_claims_rep = embedding_model.encode(data_input['claims_rep'] ,device='cuda')
    
    
    #%% [backup] matrix 생성
    
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