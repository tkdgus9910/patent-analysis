# -*- coding: utf-8 -*-
"""
Created on Mon May 27 20:20:37 2024

@author: tmlab
"""




if __name__ == '__main__':
    
    import os
    import sys
    import pandas as pd
    import numpy as np     
    import warnings
    import pickle 
    from collections import Counter
    from openai import OpenAI
    client = OpenAI()
    
    
    #%% data load
    
    warnings.filterwarnings("ignore")
    
    # 서브모듈 경로 설정
    directory = os.path.dirname(os.path.abspath(__file__))
    directory = directory.replace("\\", "/") # window
    sys.path.append(directory+'/submodule')
    import preprocess

    # 위즈도메인 csv 파일이 존재하는 디렉토리 설정
    directory = os.environ['directory_path']
    directory += '열관리/'
    
    with open(directory + 'output/save2.pkl' ,"rb") as f :
        data = pickle.load(f)
        
    #%% incremetnal clustering
    
    data['time_stamp'] = data['year_application'].apply(lambda x : int((x-2010)/2))
    
    data = data.loc[data['domain'].apply(lambda x : len(x) == 1), : ].reset_index(drop = 1) # 3386 # 중복 filtering
    data = data.loc[data['HM'].apply(lambda x : len(x) != 0), : ].reset_index(drop = 1) # 3386 # 중복 filtering
    data = data.loc[data['year_application'].apply(lambda x : x <= 2022), : ].reset_index(drop = 1) # 3386 # 중복 filtering
    
    data_exploded = data.explode('HM', ignore_index = 1)
    data_exploded['domain'] = data_exploded['domain'].apply(lambda x : x[0])
    
    data_exploded = data_exploded.sort_values(by = ['year_application']).reset_index(drop = 1)
    
    #%% load bertopic module
    
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
    from sklearn.metrics import silhouette_score
    from bertopic.vectorizers import OnlineCountVectorizer
    
    from river import cluster
    import re
    
    class Bertopic_wrapper :
        """Super Class"""
        
        def __init__(self) :
            # Step 1 - Extract embeddings
            self.embedding_model = SentenceTransformer('AI-Growth-Lab/PatentSBERTa')
            
            # Step 2 - Reduce dimensionality
            self.empty_dimensionality_model = BaseDimensionalityReduction() # 차원축소 추가수행 X
            
            # Step 4 - Tokenize topics
            custom_stopwords = ['describe', 'described', 'apparatus', 'apparatuses', 
                                'include' , 'includes', 'consists', 'consist',
                                'features', 'feature', 'device', 'devices',
                                'allows', 'allow', 'system', 'systems',
                                'invention', 'inventions', 'include' , 'includes',
                                'use', 'used', 'method', 'methods',
                                'comprise', 'comprises', 'describe', 'described'
                                'said','various','section','main']
            
            self.stop_words = list(_stop_words.ENGLISH_STOP_WORDS.union(custom_stopwords))
            
            self.vectorizer_model = CountVectorizer(stop_words= self.stop_words,
                                                # min_df = 0.0001,
                                               # max_df = 0.5,
                                               ngram_range = (1,1),
                                               preprocessor=lambda x: re.sub(r'\d+', '', x).lower().strip())
            
            # Step 5 - Create topic representation
            self.ctfidf_model = ClassTfidfTransformer()
            
            # Step 6 - (Optional) Fine-tune topic representations with 
            self.representation_model = MaximalMarginalRelevance(diversity = 0.2) # default = 0.1
            
    class Bertopic_classic(Bertopic_wrapper) :
        
        """Sub class"""
        def __init__(self, docs, embeddings):
            self.docs = docs
            self.embeddings = embeddings 
            
            super().__init__()  # 부모 클래스의 __init__ 메서드 호출
            
            
        def fit(self, return_ , **params) :
            
            # Step 3 - Cluster reduced embeddings
            cluster_model = HDBSCAN(**params,
                                    metric='euclidean',
                                    gen_min_span_tree = 1,
                                    # min_samples = 1,        
                                    prediction_data=True)
            
            if return_ == 'just_performance' :
                # Fit BERTopic without actually performing any dimensionality reduction
                model = cluster_model
                model.fit(self.embeddings)
                return(model)
            
            else : 
                model = BERTopic(
                  # Pipeline models
                  embedding_model = self.embedding_model,
                  umap_model = self.empty_dimensionality_model,
                  hdbscan_model = cluster_model,
                  vectorizer_model = self.vectorizer_model,
                  ctfidf_model = self.ctfidf_model,     
                  representation_model = self.representation_model,
                  calculate_probabilities=True,
                  # Hyperparameters
                  top_n_words=10,
                  # verbose=True,
                  # nr_topics = 30,
                )
                
                model.fit(self.docs, self.embeddings)                    
                return(model)

    
    def evaluating_cls(model, embeddings, labels, method) :

        if method == 'silhouette' :
            # outlier 제외
            non_outlier_mask = labels != -1
            if non_outlier_mask != 1 : 
                valid_embeddings = embeddings[non_outlier_mask]
                valid_labels = model.labels_[non_outlier_mask]
            else : 
                valid_embeddings = embeddings
                valid_labels = labels
            
            score = silhouette_score(valid_embeddings, valid_labels)
        
        if method == 'DBCV' : 
            score = model.relative_validity_ #hdbscan
        
        return(score)
    
    def finding_hp(docs, embeddings, K, cs_method)  : 
        
        DBCV_results = [] # Density-based clustering validation
        silhouette_results = [] 
        outliers_counts = []
        cluster_N_counts = []
        max_cluster_sizes = []
        
        for min_cluster_size in K : 
            
            bertopic_1 = Bertopic_classic(docs, embeddings)
            
            model = bertopic_1.fit("just_performance",
                                   min_cluster_size = min_cluster_size, 
                                   cluster_selection_method = cs_method, )
            
            DBCV_results.append(model.relative_validity_)
            
            # outlier 제외
            labels = model.labels_
            non_outlier_mask = labels != -1
            valid_embeddings = embeddings[non_outlier_mask]
            valid_topics = model.labels_[non_outlier_mask]
            
            score = silhouette_score(valid_embeddings, valid_topics)
            silhouette_results.append(score)
            
            # CPs.append(np.mean(topic_model.cluster_persistence_))
            
            cluster_N_counts.append(len(set(model.labels_)))
            outliers_counts.append(np.count_nonzero(model.labels_ == -1))
            max_cluster_sizes.append(np.count_nonzero(model.labels_ == 0))
            
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
        
    
    
    #%% 누적 time_stamp
    
    # time_stamps = list(data_input['time_stamp'])
    embedding_model = SentenceTransformer('AI-Growth-Lab/PatentSBERTa')
    results_time_stamp = {}
    results_time_stamp['time'] = list(range(2010,2023,2))
    results_time_stamp['model'] = []
    results_time_stamp['model_merged'] = []
    
    results_time_stamp['docs_cumulative'] = []
    results_time_stamp['topic_info'] = []
    results_time_stamp['document_info'] = []
    # results_time_stamp['topic_info'] = []
    
    umap_model = UMAP(n_neighbors=15, 
                      n_components = 5, 
                      min_dist=0.0, 
                      metric='euclidean', 
                      random_state=42)
    
    docs_cumulative = []
    
    for time_stamp in range(0,7) : 
        
        data_input = data_exploded.loc[data_exploded['time_stamp'].apply(lambda x : x == time_stamp ), :].reset_index(drop = 1)
        docs = list(data_input['HM']) 
        
        embeddings_docs = embedding_model.encode(docs ,device='cuda')
        embeddings_input = umap_model.fit_transform(embeddings_docs) 
        
        # finding_hp(docs, embeddings_input, range(5,101,5), "leaf")
        
        bertopic_1 = Bertopic_classic(docs, embeddings_input)
        model = bertopic_1.fit("final",
                               min_cluster_size=15,
                               cluster_selection_method= 'eom')
        
        results_time_stamp['model'].append(model)
        
        docs_cumulative = docs_cumulative + docs
        
        results_time_stamp['docs_cumulative'].append(docs_cumulative)
        
        model_list = results_time_stamp['model']
        model_merged = BERTopic.merge_models(model_list, min_similarity=0.95)    
        results_time_stamp['model_merged'].append(model_merged)
        result_topic_info = model_merged.get_topic_info()
        result_document_info = model_merged.get_document_info(docs_cumulative) # 40개의 도메인
    
        results_time_stamp['topic_info'].append(result_topic_info)
        results_time_stamp['document_info'].append(result_document_info)
        
        print("time : ", time_stamp, " topic_size : ", len(result_topic_info))
    
    
    #%% cross-table
    
    data_exploded['topic'] = result_document_info['Topic']
    
    contingency_table = pd.crosstab(data_exploded['topic'], data_exploded['domain'])
    
    model_merged.update_topics(docs = docs_cumulative,
                               vectorizer_model= bertopic_1.vectorizer_model)
    result_topic_info = model_merged.get_topic_info()
    
    #%%
    
    filtering_rows = contingency_table[contingency_table.sum(axis = 1) < 5].index.tolist()
    
    proportional_by_row = contingency_table.div(contingency_table.sum(axis = 1), axis = 0)
    rows_greater_than_75 = proportional_by_row.max(axis=1) > 0.8
    filtering_rows.extend(proportional_by_row[rows_greater_than_75].index.tolist()) 
    filtering_rows = list(set(filtering_rows))
    
    
    #%%  filtering results
    
    result_dynamic_filtered = {}
    result_dynamic_generation = {}
    topics = []
    
    for k, v in result_dynamic.items() : 
        v = v.loc[~v["Topic"].isin(filtering_rows)]
        result_dynamic_filtered[k] = v
        
        if k != 2010 : 
            topics = list(result_dynamic_filtered[k-2]["Topic"])
            v = v.loc[v["Topic"].isin(topics)]
            result_dynamic_generation[k] = v