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
    from sklearn.metrics import silhouette_score
    from bertopic.vectorizers import OnlineCountVectorizer
    
    from river import cluster
    
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
            self.representation_model = MaximalMarginalRelevance(diversity = 0.3) # default = 0.1
        
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
                                    min_samples = 1,        
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
    
    
    from river import stream
    from river import cluster
    
    class River:
        
        def __init__(self, model):
            self.model = model

        def partial_fit(self, umap_embeddings):
            for umap_embedding, _ in stream.iter_array(umap_embeddings):
                self.model = self.model.learn_one(umap_embedding)

            labels = []
            for umap_embedding, _ in stream.iter_array(umap_embeddings):
                label = self.model.predict_one(umap_embedding)
                labels.append(label)

            self.labels_ = labels
            return self
    
    
    class Bertopic_incremental(Bertopic_wrapper) :
        
        """Sub class"""
        def __init__(self, docs, embeddings, periods):
            
            super().__init__()  # 부모 클래스의 __init__ 메서드 호출
            self.vectorizer_model = OnlineCountVectorizer(stop_words= self.stop_words, 
                                                     # decay = 0.1,
                                                     delete_min_df = 3,
                                                     ngram_range = (1,1),
                                                     preprocessor=lambda x: re.sub(r'\d+', '', x).lower().strip())
            
            self.docs = docs
            self.embeddings = embeddings 
            self.periods = periods # list
            
            
        def fit(self, return_, **params) :
            
            # Step 3 - Cluster reduced embeddings
            cluster_model = River(cluster.DBSTREAM(**params)) # 최적
            topics = []
            
            if return_ == 'just_performance' :
                # Fit BERTopic without actually performing any dimensionality reduction
                model = cluster_model
                
                for period in sorted(list(set(self.periods))):
                    indices = list(filter(lambda x: self.periods[x] == period, range(len(self.periods))))
                    embeddings_part = [self.embeddings[i] for i in indices]
                    # periods == 0
                    model.partial_fit(embeddings_part)
                    topics.extend(model.labels_)
                    
                    print(period, " complete")
                    
            
            else : 
                
                # Step 3 - Cluster reduced embeddings
                cluster_model = River(cluster.DBSTREAM(**params)) # 최적
                
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
                )
                
                for period in sorted(list(set(self.periods))):
                    indices = list(filter(lambda x: self.periods[x] == period, range(len(self.periods))))
                    embeddings_part = self.embeddings[indices]
                    docs_part = [self.docs[i] for i in indices]
                    # periods == 0
                    model.partial_fit(documents = docs_part, embeddings = embeddings_part)
                    topics.extend(model.topics_)
                    
                    print(period, " complete")
                
            return(model, topics)
    
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
        
        
    
    #%% data load and setting
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
    
    data_input = data_preprocessed.loc[data_preprocessed["file"] == "quantum.csv", :].reset_index(drop = 1)
    

    #%%  case 1. text + hdbscan
    
    # test
    docs = list(data_input['TA']) 
    embedding_model = SentenceTransformer('AI-Growth-Lab/PatentSBERTa')
    embeddings_docs = embedding_model.encode(docs ,device='cuda')
    
    umap_model = UMAP(n_neighbors=15, 
                      n_components = 10, 
                      min_dist=0.0, 
                      metric='euclidean', 
                      random_state=42)
    
    embeddings_input = umap_model.fit_transform(embeddings_docs) 
    
    # 파라미터 탐색
    finding_hp(docs, embeddings_input, range(5,101,5), 'eom', )
     
    # 결과 도출 
    bertopic_1 = Bertopic_classic(docs, embeddings_input)
    model = bertopic_1.fit(15,'eom', "final")
    result_topic_info = model.get_topic_info()
    result_document_info = model.get_document_info(docs) # 40개의 도메인
    
    
    #%% case 2. text + CPC + hdbscan
    
    docs = list(data_input['TA']) 
    embeddings_docs = embedding_model.encode(docs ,device='cuda')
    
    umap_model = UMAP(n_neighbors=15, 
                      n_components = 10, 
                      min_dist=0.0, 
                      metric='euclidean', 
                      random_state=42)
    
    embeddings_docs = umap_model.fit_transform(embeddings_docs) 
    
    # cpc_mg
    res = pd.DataFrame([Counter(x) for x in data_input['CPC_sc']]).fillna(0) # cpc_mg
    
    umap_model = UMAP(n_neighbors=15, 
                      n_components = 10, 
                      min_dist=0.0,
                      metric = 'cosine', random_state=42)
    
    embeddings_cpc = umap_model.fit_transform(res)
    
    embeddings_input = np.concatenate((embeddings_input, embeddings_cpc), axis = 1)
    
    # 파라미터 탐색
    finding_hp(docs, embeddings_input, range(5,101,5), 'eom', )
     
    # 결과 도출 
    bertopic_1 = Bertopic_classic(docs, embeddings_input)
    model = bertopic_1.fit(40,'eom', "final")
    result_topic_info = model.get_topic_info()
    result_document_info = model.get_document_info(docs) # 40개의 도메인
    # bertopic_parameter_tuning(range(5,101,5), 'eom', docs, embeddings_input)
    
    #%% case 3. incremental - merging model
    
    # time_stamps = list(data_input['time_stamp'])
    embedding_model = SentenceTransformer('AI-Growth-Lab/PatentSBERTa')
    
    data_input = data_preprocessed.loc[data_preprocessed["file"] == "quantum.csv", :].reset_index(drop = 1)
    data_input = data_input.loc[data_input['year_application'] <= 2020, :]
    data_input = data_input.loc[data_input['year_application'] >= 2010, :].reset_index(drop = 1)
    data_input['time_stamp'] = data_input['year_application'].apply(lambda x : int((x-2010)/2))
    
    results_time_stamp = {}
    results_time_stamp['time'] = list(range(2010,2020,2))
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
    
    for time_stamp in range(0,6) : 
        
        data_temp = data_input.loc[data_input['time_stamp'].apply(lambda x : x == time_stamp ), :].reset_index(drop = 1)
        docs = list(data_temp['abstract']) 
        
        embeddings_docs = embedding_model.encode(docs ,device='cuda')
        embeddings_input = umap_model.fit_transform(embeddings_docs) 
        
        # finding_hp(docs, embeddings_input, range(5,101,5), "eom")
        
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
    
    #%% case 3-2. incremental-dbstream
    
    # docs_time_series = []
    data_input = data_preprocessed.loc[data_preprocessed["file"] == "quantum.csv", :].reset_index(drop = 1)
    data_input = data_input.loc[data_input['year_application']<= 2012, :]
    docs = list(data_input['TA']) 
    embedding_model = SentenceTransformer('AI-Growth-Lab/PatentSBERTa')
    embeddings_docs = embedding_model.encode(docs ,device='cuda')
    
    umap_model = UMAP(n_neighbors=15, 
                      n_components = 10, 
                      min_dist=0.0, 
                      metric='euclidean', 
                      random_state=42)
    
    embeddings_input = umap_model.fit_transform(embeddings_docs) 
    
    periods = list(data_input['time_stamp'])

    bertopic_2 = Bertopic_incremental(docs, embeddings_input, periods)
    
    
    model, topics = bertopic_2.fit("final", clustering_threshold = 3)
    result_topic_info = model.get_topic_info()
    
    
    
    #%% case 3-3. finding best model
    
    # evaluating_cls(model, embeddings_input, topics, method = "silhouette")

    from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
    
    data_input = data_preprocessed.loc[data_preprocessed["file"] == "quantum.csv", :].reset_index(drop = 1)
    
    data_input['time_stamp'] = data_input['year_application'].apply(lambda x : int((x-2000)/3))
    data_input = data_input.loc[data_input['time_stamp']<= 5, :]
    
    docs = list(data_input['TA']) 
    embedding_model = SentenceTransformer('AI-Growth-Lab/PatentSBERTa')
    embeddings_docs = embedding_model.encode(docs ,device='cuda')
    
    time_stamps = list(data_input['time_stamp'])
    
    umap_model = UMAP(n_neighbors=15, 
                      n_components = 10, 
                      min_dist=0.0, 
                      metric='euclidean', 
                      random_state=42)
    
    embeddings_input = umap_model.fit_transform(embeddings_docs) 
    
    bertopic_2 = Bertopic_incremental(docs, embeddings_input, time_stamps)
    
    # Define the hyperparameter space
    
    space = {'clustering_threshold' : hp.uniform('clustering_threshold', 0.5, 5), #default 1.0, 커지면 마이크로 클러스터 생성 x
            # 'fading_factor' : hp.uniform('fading_factor', 0.005, 0.015), 
            # 'intersection_factor' : hp.uniform('intersection_factor', 0.15, 0.6), #default 1.0, 커지면 마이크로 클러스터 생성 x
            # 'minimum_weight' : hp.uniform('minimum_weight', 0.5, 1.5) #default 1.0, 커지면 마이크로 클러스터 생성 x
            }
    
    # Run the optimizer
    trials = Trials()
    
    def objective(params):
        
        model, topics = bertopic_2.fit("just_performance", **params)
        
        score = evaluating_cls(model, embeddings_input, topics, method = "silhouette")
        return {'loss': -score, 'status': STATUS_OK} 
        
    best = fmin(objective,
                space=space,
                algo=tpe.suggest,
                max_evals = 10,
                trials=trials)
    