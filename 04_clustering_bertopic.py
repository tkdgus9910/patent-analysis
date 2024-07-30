# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 15:25:05 2024

@author: tmlab
"""







if __name__ == '__main__':
    
    import os
    import sys
    import pandas as pd
    import numpy as np     
    import warnings
    import pickle 
    import re
    from collections import Counter
    
    from sentence_transformers import SentenceTransformer
    from umap import UMAP
    import matplotlib.pyplot as plt
    from hdbscan import HDBSCAN
    from sklearn.feature_extraction import _stop_words
    from sklearn.feature_extraction.text import CountVectorizer
    
    from bertopic.representation import MaximalMarginalRelevance
    from bertopic import BERTopic
    from bertopic.dimensionality import BaseDimensionalityReduction
    

    from bertopic.representation import KeyBERTInspired
    from bertopic.vectorizers import ClassTfidfTransformer
    
    #%%
    
    def bertopic_wrapper(min_cluster_size, #hdbscan 파라미터
                         cs_method, #hdbscan 파라미터
                         docs, # 학습할 문서
                         embeddings, # 학습할 문서의 embedding vector
                         return_ #결과 반환여부
                         ) :
        
        # Step 1 - Extract embeddings
        embedding_model = SentenceTransformer('AI-Growth-Lab/PatentSBERTa')
        
        # Step 2 - Reduce dimensionality
        empty_dimensionality_model = BaseDimensionalityReduction() # 차원축소 추가수행 X
        
        # Step 3 - Cluster reduced embeddings
        hdbscan_model = HDBSCAN(min_cluster_size= min_cluster_size,
                                metric='euclidean',
                                cluster_selection_method= cs_method,
                                gen_min_span_tree = 1,
                                min_samples = 1,        
                                prediction_data=True)
        
        # Step 4 - Tokenize topics
        custom_stopwords = ['describe', 'described', 'apparatus', 'apparatuses', 
                            'include' , 'includes', 'consists', 'consist',
                            'features', 'feature', 'device', 'devices',
                            'allows', 'allow', 'system', 'systems',
                            'invention', 'inventions', 'include' , 'includes',
                            'use', 'used', 'method', 'methods',
                            'comprise', 'comprises', 'describe', 'described'
                            'said','various','section','main']
        
        stop_words = list(_stop_words.ENGLISH_STOP_WORDS.union(custom_stopwords))
        
        vectorizer_model = CountVectorizer(stop_words= stop_words,
                                           # min_df = 0.0001,
                                           # max_df = 0.5,
                                           ngram_range = (1,2),
                                           preprocessor=lambda x: re.sub(r'\d+', '', x).lower().strip())
        
        # Step 5 - Create topic representation
        ctfidf_model = ClassTfidfTransformer()
        
        # Step 6 - (Optional) Fine-tune topic representations with 
        # a `bertopic.representation` model
        # representation_model = KeyBERTInspired()
        representation_model = MaximalMarginalRelevance(diversity = 0.2) # default = 0.1
        
        if return_ == 'just_performance' :
            # Fit BERTopic without actually performing any dimensionality reduction
            cluster_model = hdbscan_model
            cluster_model.fit(embeddings)
            
            return(cluster_model)
        
        else : 
            
            topic_model = BERTopic(
              # Pipeline models
              embedding_model=embedding_model,
              umap_model=empty_dimensionality_model,
              hdbscan_model= hdbscan_model,
              vectorizer_model=vectorizer_model,
              ctfidf_model=ctfidf_model,     
              representation_model=representation_model,
              calculate_probabilities=True,
              # Hyperparameters
              top_n_words=10,
              # verbose=True,
              # nr_topics = 30,
            )
            
            topic_model.fit(docs, embeddings)    
            
            # Train model
            return(topic_model)
    
    
    from sklearn.metrics import silhouette_score
    
    def bertopic_parameter_tuning(K, 
                                  cs_method,
                                  docs, 
                                  embeddings, ) : 
        """
        K = range(5, 101, 5)
        method = 'leaf', 'eom'
        """
        
        DBCV_results = [] # Density-based clustering validation
        silhouette_results = [] 
        
        outliers_counts = []
        cluster_N_counts = []
        max_cluster_sizes = []
        
        for min_cluster_size in K : 
            
            topic_model = bertopic_wrapper(
                min_cluster_size,  # hdbscan 파라미터
                cs_method,  # hdbscan 파라미터
                docs, # 학습할 문서
                embeddings, # 학습할 문서의 embedding vector
                return_  = 'just_performance' # 결과 반환여부
                )
            
            DBCV_results.append(topic_model.relative_validity_)
            
            # outlier 제외
            labels = topic_model.labels_
            non_outlier_mask = labels != -1
            valid_embeddings = embeddings[non_outlier_mask]
            valid_topics = topic_model.labels_[non_outlier_mask]
            
            
            score = silhouette_score(valid_embeddings, valid_topics)
            silhouette_results.append(score)
            
            # CPs.append(np.mean(topic_model.cluster_persistence_))
            
            cluster_N_counts.append(len(set(topic_model.labels_)))
            outliers_counts.append(np.count_nonzero(topic_model.labels_ == -1))
            max_cluster_sizes.append(np.count_nonzero(topic_model.labels_ == 0))
            
            print(min_cluster_size)
        
        
        # 시각화-DBCV
        plt.figure(figsize=(12,8))
        plt.subplot(311)
        plt.plot(K, DBCV_results, 'bx-')
        # plt.xlabel('min_cluister_size')
        plt.ylabel('DBCVs', fontsize=15)
        
        # 시각화-DBCV    
        plt.subplot(312)
        plt.plot(K, silhouette_results, 'gx-')
        plt.ylabel('Silhouettes', fontsize=15)
        
        # 시각화-Outlier
        plt.subplot(313)
        outliers_ratios = [round(i / len(docs), 3) for i in outliers_counts]
        plt.plot(K, outliers_ratios, 'rx-')
        # plt.xlabel('min_cluister_size', fontsize=15)
        plt.xlabel('min_cluister_size', fontsize=15)
        plt.ylabel('outliers_ratio', fontsize=15)
        
        # 시각화
        plt.figure(figsize=(12,8))
        plt.subplot(211)
        plt.plot(K, cluster_N_counts, 'rx-')
        # plt.xlabel('min_cluister_size')
        plt.ylabel('cluster_counts', fontsize=15)
        
        # 시각화
        plt.subplot(212)
        max_cluster_sizes = [round(i / len(docs), 3) for i in max_cluster_sizes]
        plt.plot(K, max_cluster_sizes, 'gx-')
        plt.xlabel('min_cluister_size', fontsize=15)
        plt.ylabel('biggest_cluster_ratio', fontsize=15)
        plt.show()
        

        
        
    #%%
    warnings.filterwarnings("ignore")
    
    # 서브모듈 경로 설정
    directory = os.path.dirname(os.path.abspath(__file__))
    directory = directory.replace("\\", "/") # window
    sys.path.append(directory+'/submodule')
            
    # 위즈도메인 csv 파일이 존재하는 디렉토리 설정
    directory = os.environ['directory_path']
    directory += 'LLM/'
    
    file_list = os.listdir(directory)
    
    # 데이터 로드 
    with open(directory + 'data_preprocessed.pkl', 'rb') as file:  
        data_preprocessed = pickle.load(file)  
    
    
    #%% case 1. only text
    
    embedding_model = SentenceTransformer('AI-Growth-Lab/PatentSBERTa')
    
    data_input = data_preprocessed.loc[data_preprocessed["file"] == "quantum.csv", :].reset_index(drop = 1)
    docs = list(data_input['TA']) 
    
    embeddings_docs = embedding_model.encode(docs ,device='cuda')
    umap_model = UMAP(n_neighbors=15, 
                      n_components = 10, 
                      min_dist=0.0, 
                      metric='euclidean', 
                      random_state=42)
    embeddings_input = umap_model.fit_transform(embeddings_docs) 
    
    bertopic_parameter_tuning(range(5,101,5), 
                              'eom', 
                              docs, 
                              embeddings_input)
    
    
    topic_model1 = bertopic_wrapper(15, 'eom',
                                    docs, 
                                    embeddings_input,
                                    return_ = 'final_result')
    
    result_topic_info = topic_model1.get_topic_info()
    result_document_info = topic_model1.get_document_info(docs) # 40개의 도메인
    
    #%% case 2. with CPC
    
    # cpc_mg
    res = pd.DataFrame([Counter(x) for x in data_input['CPC_sc']]).fillna(0) # cpc_mg
    
    umap_model = UMAP(n_neighbors=15, 
                      n_components = 10, 
                      min_dist=0.0,
                      metric = 'cosine', random_state=42)
    
    embeddings_cpc = umap_model.fit_transform(res)
    
    embeddings_input = np.concatenate((embeddings_input, embeddings_cpc), axis = 1)
    
    bertopic_parameter_tuning(range(5,101,5), 'eom', docs, embeddings_input)
    
    #%% case 3. get topic
    
    topic_model1 = bertopic_wrapper(35, 
                                    'eom', 
                                    docs, 
                                    embeddings_input,
                                    return_ = 'final_result')
    
    result_topic_info = topic_model1.get_topic_info()
    result_document_info = topic_model1.get_document_info(docs) # 40개의 도메인