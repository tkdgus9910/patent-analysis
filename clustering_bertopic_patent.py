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
    from copy import copy
    import re
    from collections import Counter
    
    from sentence_transformers import SentenceTransformer
    from umap import UMAP
    import matplotlib.pyplot as plt
    from hdbscan import HDBSCAN
    from sklearn.feature_extraction import text 
      
    from sklearn.feature_extraction.text import CountVectorizer
    
    from bertopic.representation import MaximalMarginalRelevance
    from bertopic import BERTopic
    from bertopic.dimensionality import BaseDimensionalityReduction
    from sklearn.feature_extraction import _stop_words
    
    def bertopic_wrapper(min_cluister_size, docs, embeddings, cs_method,return_) :
        
        representation_model = MaximalMarginalRelevance(diversity = 0.3) # default = 0.1
        empty_dimensionality_model = BaseDimensionalityReduction()
        
        custom_stopwords = ['described', 'apparatus', 'include' , 'includes', 'consists', 'features', 
                            'feature', 'device', 'allows', 'allow', 'use', 'method', 'comprises', 'various', 'section', 'said']
        stop_words = list(_stop_words.ENGLISH_STOP_WORDS.union(custom_stopwords))
        
        vectorizer_model = CountVectorizer(stop_words= stop_words,
                                           min_df = 0.0001,
                                           max_df = 0.5,
                                           ngram_range = (1,2),
                                           preprocessor=lambda x: re.sub(r'\d+', '', x).lower().strip())
        
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
              # nr_topics = 30,
            )
            
            topic_model.fit(docs, embeddings)    
            
            # Train model
            return(topic_model)
    
    
    def bertopic_parameter_tuning(K, docs_input, embeddings_input, method) : 
        """
        K = range(5, 101, 5)
        method = 'leaf', 'eom'
        
        """
        DBCVs = [] # Density-based clustering validation
        CPs = [] 
        outliers_counts = []
        cluster_N_counts = []
        max_cluster_sizes = []
        
        for size in K : 
            
            topic_model = bertopic_wrapper(size, 
                                           docs_input, 
                                           embeddings_input, method, return_ = 'just_performance')
            
            DBCVs.append(topic_model.relative_validity_)
            CPs.append(np.mean(topic_model.cluster_persistence_))
            
            cluster_N_counts.append(len(set(topic_model.labels_)))
            outliers_counts.append(np.count_nonzero(topic_model.labels_ == -1))
            max_cluster_sizes.append(np.count_nonzero(topic_model.labels_ == 0))
            
            print(size)
        
        
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
        
        
        
    #%% phase 0. load data
    warnings.filterwarnings("ignore")
    
    # 서브모듈 경로 설정
    directory = os.path.dirname(os.path.abspath(__file__))
    directory = directory.replace("\\", "/") # window
    sys.path.append(directory+'/submodule')
            
    # 위즈도메인 csv 파일이 존재하는 디렉토리 설정
    directory = os.environ['directory_path']
    directory += '우주/'
    
    file_list = os.listdir(directory)
    
    # 데이터 로드 
    with open(directory + 'data_background.pkl', 'rb') as file:  
        data_preprocessed = pickle.load(file)  
        
    
    #%% phase 1. only text
    
    embedding_model = SentenceTransformer('AI-Growth-Lab/PatentSBERTa')
    
    docs = list(data_preprocessed['TA']) 
    docs = list(data_preprocessed['TAF']) #best? 
    
    embeddings_docs = embedding_model.encode(docs ,device='cuda')
    
    umap_model = UMAP(n_neighbors=15, n_components = 10, min_dist=0.0, metric='cosine', random_state=42)
    embeddings_input = umap_model.fit_transform(embeddings_docs) 
    
    bertopic_parameter_tuning(range(5,101,5), docs, embeddings_input, method = 'eom')
    
    #%% phase 2. get best model
    
    topic_model1 = bertopic_wrapper(65, 
                                    docs, 
                                    embeddings_input,
                                    'eom', 
                                    return_ = 'final_result')
    
    result_topic_info = topic_model1.get_topic_info()
    result_document_info = topic_model1.get_document_info(docs) # 40개의 도메인