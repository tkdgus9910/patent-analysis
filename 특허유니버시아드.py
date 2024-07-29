# -*- coding: utf-8 -*-
"""
Created on Sat Jun 29 23:08:59 2024

@author: tmlab
"""
# from copy import copy 
# data_old = copy(data)

#%% 1. 데이터 로드 및 세팅

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
    
    # data load
    
    warnings.filterwarnings("ignore")
    
    # 서브모듈 경로 설정
    directory = os.path.dirname(os.path.abspath(__file__))
    directory = directory.replace("\\", "/") # window
    sys.path.append(directory+'/submodule')
    import preprocess
    
    
    # 위즈도메인 csv 파일이 존재하는 디렉토리 설정
    directory = "D:/OneDrive/특허유니버시아드/데이터/"
    
    file_list = os.listdir(directory)
    file_list = [i for i in file_list if ".csv" in i ]
    
    data = pd.DataFrame()
    # domain_dict = {}
    
    for file in file_list : 
        
        print(file)
        data_temp = pd.read_csv(directory + file, skiprows= 4)
        data_temp = preprocess.wisdomain_prep(data_temp) 
        data_temp['domain'] = file.split(".")[0]
        
        # domain_dict[file.split('.')[0]] = list(data_temp['id_wisdomain'])
        data = pd.concat([data, data_temp], axis = 0).reset_index(drop = 1)
        
    
    data = data.drop_duplicates(subset= ['id_wisdomain']).reset_index(drop = 1) # 중복제거 
    
    #%% 1-2. PIM 데이터 문제 해결
    
    # pim 추가 로드
    
    # 위즈도메인 csv 파일이 존재하는 디렉토리 설정
    directory = "D:/OneDrive/특허유니버시아드/데이터/PIM_추가수집/"
    
    file_list = os.listdir(directory)
    file_list = [i for i in file_list if ".csv" in i ]
    
    data_pim = pd.DataFrame()
    
    for file in file_list : 
        
        data_temp = pd.read_csv(directory + file, skiprows= 4)
        data_temp = preprocess.wisdomain_prep(data_temp) 
        data_temp['country'] = data_temp['id_wisdomain'].apply(lambda x : x[0:2])
        data_pim = pd.concat([data_pim, data_temp], axis = 0).reset_index(drop = 1)
    
    import re 
    from collections import Counter
    
    # Define the regex pattern
    pattern = re.compile(r'(?:processor|processing|process)\s+in\s+memory|(?:프로세싱|프로세서|프로세스)\s+인\s+메모리', re.IGNORECASE)
    # pattern = re.compile(r'processing in memory', re.IGNORECASE)

    def check_match(s):
        return 1 if pattern.search(s) else 0
    
    data_pim['test'] = 0
    
    data_pim['test'] = data_pim.apply(lambda x : check_match(x.title_) if (x.test == 0) and (str(x.title_) != "nan") else x.test, axis = 1)
    
    data_pim['test'] = data_pim.apply(lambda x : check_match(x.title) if (x.test == 0) and (str(x.title) != "nan") else x.test, axis = 1)
    
    data_pim['test'] = data_pim.apply(lambda x : check_match(x.abstract) if (x.test == 0) and (str(x.abstract) != "nan") else x.test, axis = 1)
    data_pim['test'] = data_pim.apply(lambda x : check_match(x.abstract_) if (x.test == 0) and (str(x.abstract_) != "nan") else x.test, axis = 1)
    data_pim['test'] = data_pim.apply(lambda x : check_match(x.claims_rep) if (x.test == 0) and (str(x.claims_rep) != "nan") else x.test, axis = 1)
    
    #%% 중간결과물 저장
    
    temp = data_pim.loc[data_pim['test'] == 1, :].reset_index(drop = 1)
    
    temp['domain'] = "PIM_일반"
    temp.to_pickle(directory + "PIM_preprocssed.pkl")
    
    #%% 1-3. 기존 PIM 제거 및 PIM 대체
    
    # 1. 기존 PIM 제거 코드
    data = data.loc[data['domain'] != 'PIM_일반', : ]
    
    directory = "D:/OneDrive/특허유니버시아드/데이터/output/"
    
    # 데이터 로드
    with open(directory + 'PIM_preprocssed.pkl', 'rb') as f:
    	data_pim = pickle.load(f)
    
    data = pd.concat([data, data_pim], axis = 0)
    
    data = data.drop_duplicates(subset= ['id_wisdomain']).reset_index(drop = 1) # 중복제거 
    
    #%% 1-4 EDA 1. 국가 별 시계열 출원 동향 
    
    # data_input['domain'] = data_input['domain'].apply(lambda x : x.split("_")[0])
    
    # temp = data_input.groupby(['country',"domain"]).size().unstack(fill_value=0)
    
    from copy import copy
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    
    data_input = pd.concat([data_us, data_korean, data_chinese], axis = 0 )
    
    data_input = copy(data_input.loc[data_input['year_application'] <= 2023, :].reset_index(drop= 1))
    data_input = copy(data_input.loc[data_input['year_application'] >= 2005, :].reset_index(drop= 1))
    
    year_counts = data_input.groupby('country')['year_application'].value_counts().sort_index().reset_index()
    year_counts['year_application'] = year_counts['year_application'].apply(lambda x : str(x))
    # year_counts = data_input['year_application'].value_counts().sort_index()

    total_measurements = year_counts.groupby('year_application')['count'].sum().reset_index()
    total_measurements['country'] = 'Total'
    
    # year_counts = pd.concat([year_counts, total_measurements], ignore_index=True)
    
    plt.figure(figsize=(12, 6))
    sns.lineplot(data = year_counts,
                 x='year_application', 
                 y='count', 
                 hue = 'country',
                 marker='o',)
    
    sns.barplot(data=total_measurements, x='year_application', 
                y='count', 
                hue='country', palette='Greys')
    
    
    plt.xlabel('Year',fontsize = 15)
    plt.ylabel('Annual Counts',fontsize = 15)
    plt.title('Patent Application Trend',fontsize = 15)
    
    plt.axvline(x = '2022', color = 'r', label = '18-month publication')
    
    # x축 ticks 조정
    
    # The arrow starts at (2,10) and ends at (8,10)
    
    # Annotate each point with its frequency
    for i, freq in enumerate(total_measurements['count']):
        
        plt.text(i, freq + 5, str(freq), ha='center')  # Adjusting y-offset for better visibility
    
    plt.show()
    
    #%% 1-5 EDA 2. 기술 별 시계열 출원 동향
    
    from copy import copy
    import seaborn as sns
    import matplotlib.pyplot as plt
    import re
    
    data_input = pd.concat([data_us, data_korean, data_chinese], axis = 0 )
    data_input = copy(data_input.loc[data_input['year_application'] <= 2023, :].reset_index(drop= 1))
    data_input = copy(data_input.loc[data_input['year_application'] >= 2005, :].reset_index(drop= 1))
    
    data_input['domain'] = data_input['domain'].apply(lambda x : x.split("_")[0])
    data_input['domain'] = data_input['domain'].apply(lambda x : re.sub("뉴로모픽", "Neuromorphic", x))
    data_input['domain'] = data_input['domain'].apply(lambda x : re.sub("멤리스터", "memristor", x))
    data_input['domain'] = data_input['domain'].apply(lambda x : re.sub("일반", "general", x))
    
    year_counts = data_input.groupby('domain')['year_application'].value_counts().sort_index().reset_index()
    year_counts['year_application'] = year_counts['year_application'].apply(lambda x : str(x))
    
    year_counts = year_counts.sort_values(by = ['year_application'])

    # year_counts = pd.concat([year_counts, total_measurements], ignore_index=True)
    
    plt.figure(figsize=(12, 6))
    sns.lineplot(data = year_counts,
                 x='year_application', 
                 y='count', 
                 hue = 'domain',
                 marker='o',)
    
    
    plt.xlabel('Year',fontsize = 15)
    plt.ylabel('Annual Counts',fontsize = 15)
    plt.title('Patent Application Trend',fontsize = 15)
    
    plt.axvline(x = '2022', color = 'r', label = '18-month publication')
    
    
    # Annotate each point with its frequency
    for group in set(year_counts['domain']) : 
        if group not in ["HBM", "Neuromorphic", "NPU", "CXL"] : continue
    
        year_counts_group = year_counts.loc[year_counts['domain'] == group, :]
        
        for i, freq in enumerate(year_counts_group['count']):
            plt.text(i, freq + 5, str(freq), ha='center')  # Adjusting y-offset for better visibility
        
    plt.show()
    
    #%% 업체 별 기술 출원 동향
    
    
    from copy import copy
    import seaborn as sns
    import matplotlib.pyplot as plt
    import re
    
    data_input = pd.concat([data_us, data_korean, data_chinese], axis = 0 )
    data_input = copy(data_input.loc[data_input['year_application'] <= 2023, :].reset_index(drop= 1))
    data_input = copy(data_input.loc[data_input['year_application'] >= 2005, :].reset_index(drop= 1))
    
    data_input['domain'] = data_input['domain'].apply(lambda x : x.split("_")[0])
    data_input['domain'] = data_input['domain'].apply(lambda x : re.sub("뉴로모픽", "Neuromorphic", x))
    data_input['domain'] = data_input['domain'].apply(lambda x : re.sub("멤리스터", "memristor", x))
    data_input['domain'] = data_input['domain'].apply(lambda x : re.sub("일반", "general", x))
    
    year_counts = data_input.groupby('domain')['year_application'].value_counts().sort_index().reset_index()
    year_counts['year_application'] = year_counts['year_application'].apply(lambda x : str(x))
    
    year_counts = year_counts.sort_values(by = ['year_application'])

    # year_counts = pd.concat([year_counts, total_measurements], ignore_index=True)
    
    plt.figure(figsize=(12, 6))
    sns.lineplot(data = year_counts,
                 x='year_application', 
                 y='count', 
                 hue = 'domain',
                 marker='o',)
    
    
    plt.xlabel('Year',fontsize = 15)
    plt.ylabel('Annual Counts',fontsize = 15)
    plt.title('Patent Application Trend',fontsize = 15)
    
    plt.axvline(x = '2022', color = 'r', label = '18-month publication')
    
    
    # Annotate each point with its frequency
    for group in set(year_counts['domain']) : 
        if group not in ["HBM", "Neuromorphic", "NPU", "CXL"] : continue
    
        year_counts_group = year_counts.loc[year_counts['domain'] == group, :]
        
        for i, freq in enumerate(year_counts_group['count']):
            plt.text(i, freq + 5, str(freq), ha='center')  # Adjusting y-offset for better visibility
        
    plt.show()
    
    
    #%% 2-1. 체계도 구축 시작
    data_old = copy(data_us)
    #%%
    data_us = data.loc[data['country'] == "US", : ].reset_index(drop = 1)
    
    data_us = data_us.sort_values(by = 'date_application_strp', axis = 0)
    
    # 패밀리 제거
    data_us['id_family'] = data_us['id_family'].apply(lambda x : str(int(x)) if str(x) != 'nan' else x)
    data_us = data_us.drop_duplicates(subset=  ['id_family'], keep = 'first').reset_index(drop = 1)
    
    #%% 2-2. generating field

    
    client = OpenAI(api_key= key)
    
    field_dict = dict(zip(data_old['id_wisdomain'], data_old['field_generated']))
    
    data_us['field_generated'] = ""
    data_us['field_generated'] = data_us['id_wisdomain'].apply(lambda x : field_dict[x] if x in field_dict.keys() else "")
    
    
    instruction = "Your role is to create a field for the invention, given the title, abstract and 1st claim of the patent."
    
    for idx, row in data_us.iterrows() : 
        print(idx)
        
        if len(row["field_generated"]) == 0 :
            prompt = "Title : "
            prompt += row['title']
            prompt += "\n\nAbstract : "
            prompt += row['abstract']
            prompt += "\n\n1st claims : "
            prompt += row['claims_rep']
        
            completion = client.chat.completions.create(
                # model="ft:gpt-3.5-turbo-1106:snu:ptfunction:9eyA1SyW",
                model="ft:gpt-3.5-turbo-0125:snu:patentfield:9M9lmZnK",
                messages=[
                {"role": "user", 
                 "content": prompt},
                {"role" : "system", "content" : instruction}
                ],temperature = 0.0)
            
            data_us["field_generated"][idx] = completion.choices[0].message.content
            
    
    data_us.to_pickle(directory + 'data_us_0717.pkl')
    
    #%% 2-3. incremental clustering
    
    from sentence_transformers import SentenceTransformer
    from umap import UMAP
    import matplotlib.pyplot as plt
    from sklearn.feature_extraction import text 
    from sklearn.feature_extraction import _stop_words
    
    from bertopic.representation import MaximalMarginalRelevance
    from bertopic import BERTopic
    from bertopic.dimensionality import BaseDimensionalityReduction
    
    
    from river import stream
    from river import cluster
    from bertopic.vectorizers import ClassTfidfTransformer
    from bertopic.vectorizers import OnlineCountVectorizer
    
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

    # document embeddings
    
    embedding_model = SentenceTransformer('AI-Growth-Lab/PatentSBERTa')
    
    docs = data_us['abstract']    
    embeddings_docs = embedding_model.encode(docs, device='cuda')
    umap_model = UMAP(n_neighbors=15, n_components = 5, min_dist=0.0, metric='cosine', random_state=42)
    
    embeddings_input = umap_model.fit_transform(embeddings_docs) 
    
    data_us['embedding'] = embeddings_input.tolist()
    
    docs_time_series = []
    
    for year in range(2005, 2023, 6) :
        print(year)
        data_temp = data_us.loc[(data_us['year_application'] >= year) , :].reset_index(drop = 1)
        
        if year != 2017 : 
            data_temp = data_temp.loc[(data_temp['year_application'] < year+6) , :].reset_index(drop = 1)
        else : 
            pass
            data_temp = data_temp.loc[(data_temp['year_application'] < year+7) , :].reset_index(drop = 1)
     
        docs_time_series.append(data_temp)
    
    #%% hp tuning
    
    from sklearn.metrics import silhouette_score
    from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
    
    custom_stopwords = ['described', 'apparatus', 'include' , 'includes', 'consists', 'features', 
                        'feature', 'device', 'allows', 'allow', 'use', 'method', 'comprises', 'various', 'section', 'said',
                        'related', 'particularly', 'present', 'invention', 'device', 'devices',]
    
    stop_words = list(_stop_words.ENGLISH_STOP_WORDS.union(custom_stopwords))
    
    vectorizer_model = OnlineCountVectorizer(stop_words= stop_words, 
                                             # decay = 0.1,
                                             delete_min_df = 3,)
    
    empty_dimensionality_model = BaseDimensionalityReduction()
    
    ctfidf_model = ClassTfidfTransformer(
        # reduce_frequent_words=True, 
        # bm25_weighting=True,
        )
    
    embedding_model = SentenceTransformer('AI-Growth-Lab/PatentSBERTa')
    
    #%% # Define the objective function to minimize
    
    def objective(params):
        
        vectorizer_model = OnlineCountVectorizer(stop_words= stop_words, 
                                                 # decay = 0.1,
                                                 delete_min_df = 3,)
        
        cluster_model = River(cluster.DBSTREAM(**params)) # 최적
        
        # Prepare model
        topic_model = BERTopic(
            embedding_model=embedding_model,
            hdbscan_model=cluster_model, 
            vectorizer_model=vectorizer_model, 
            ctfidf_model=ctfidf_model,
            umap_model= empty_dimensionality_model,
            representation_model = MaximalMarginalRelevance(diversity = 0.1) # default = 0.1
        )
        
        result_dynamic = {}
        
        year = 2005
        gap = 6
        idx_save = 0 
        
        for idx, data_temp in enumerate(docs_time_series) :
            
            documents = data_temp['field_generated']
            embeddings = np.stack(data_temp['embedding'].to_numpy())
            print(year)
            
            topic_model.partial_fit(documents, embeddings = embeddings)
            
            result_topic_info = topic_model.get_topic_info()
            result_document_info = topic_model.get_document_info(documents) 
            result_dynamic[year] = result_topic_info
            
            count = len(documents)
            
            data_temp['Topic'] = result_document_info['Topic']
            docs_time_series[idx] = data_temp 
            
            idx_save += count
            year += gap
        
        # calculate ARI
        docs_time_series_cumulative = []
        
        temp = pd.DataFrame()
        
        for idx, row in enumerate(docs_time_series) : 
            # row = row.loc[~row['Topic'].isin(filtering_rows),:].reset_index(drop = 1)
            temp = pd.concat([temp, row], axis = 0)
            docs_time_series_cumulative.append(temp)    
            
        # calculate ARI
        # ari_score = adjusted_rand_score(docs_time_series_cumulative[-1]['domain'],
                                        # docs_time_series_cumulative[-1]['Topic']) # 0~1
                                        
        # valid_embeddings = docs_time_series[-1]['embedding']
        valid_embeddings = embeddings
        # valid_topics = docs_time_series[-1]['embedding']
        valid_topics = result_document_info['Topic']
        
        score = silhouette_score(valid_embeddings, valid_topics)
        
        return {'loss': -score, 'status': STATUS_OK} 
    
    #%%
    
    # Define the hyperparameter space
    
    space = {'clustering_threshold' : hp.uniform('clustering_threshold', 0.5, 1.5), #default 1.0, 커지면 마이크로 클러스터 생성 x
            'fading_factor' : hp.uniform('fading_factor', 0.005, 0.015), #default 1.0, 커지면 마이크로 클러스터 생성 x
            'intersection_factor' : hp.uniform('intersection_factor', 0.15, 0.6), #default 1.0, 커지면 마이크로 클러스터 생성 x
            'minimum_weight' : hp.uniform('minimum_weight', 0.5, 1.5)#default 1.0, 커지면 마이크로 클러스터 생성 x
            }
    
    # Run the optimizer
    trials = Trials()
    
    best = fmin(objective,
                space=space,
                algo=tpe.suggest,
                max_evals = 40,
                trials=trials)
    
    #%% best?
    
    cluster_model = River(cluster.DBSTREAM(**best)) # 최적
    
    vectorizer_model = OnlineCountVectorizer(stop_words= stop_words, 
                                             # decay = 0.1,
                                             delete_min_df = 3,)
    
    empty_dimensionality_model = BaseDimensionalityReduction()
    
    ctfidf_model = ClassTfidfTransformer(
        reduce_frequent_words=True, 
        bm25_weighting=True,)
    
    embedding_model = SentenceTransformer('AI-Growth-Lab/PatentSBERTa')
    
    # Prepare model
    topic_model = BERTopic(
        embedding_model=embedding_model,
        hdbscan_model=cluster_model, 
        vectorizer_model=vectorizer_model, 
        ctfidf_model=ctfidf_model,
        umap_model= empty_dimensionality_model,
        representation_model = MaximalMarginalRelevance(diversity = 0.1) # default = 0.1
    )
    
    result_dynamic = {}
    
    year = 2005
    gap = 6
    idx_save = 0 
    
    for idx, data_temp in enumerate(docs_time_series) :
        
        documents = data_temp['abstract']
        embeddings = np.stack(data_temp['embedding'].to_numpy())
        
        print(year)
        
        topic_model.partial_fit(documents, embeddings = embeddings)
        
        result_topic_info = topic_model.get_topic_info()
        result_document_info = topic_model.get_document_info(documents) 
        result_dynamic[idx] = result_topic_info
        
        count = len(documents)
        
        data_temp['Topic'] = result_document_info['Topic']
        docs_time_series[idx] = data_temp 
        
        idx_save += count
        year += gap
    
    # calculate ARI
    
    docs_time_series_cumulative = []
    
    temp = pd.DataFrame()
    
    for idx, row in enumerate(docs_time_series) : 
        # row = row.loc[~row['Topic'].isin(filtering_rows),:].reset_index(drop = 1)
        
        temp = pd.concat([temp, row], axis = 0)
        
        docs_time_series_cumulative.append(temp)
    
    valid_embeddings = embeddings
    
    # valid_topics = docs_time_series[-1]['embedding']
    
    valid_topics = result_document_info['Topic']
    
    score = silhouette_score(valid_embeddings, valid_topics)
    
    print(score)
    # parameter?
    temp = trials.losses()
    
    
    #%% 주제 별 대표 특허 할당
    
    for idx in range(len(docs_time_series)) : 
        
        temp = docs_time_series_cumulative[idx]
        
        temp = temp.sort_values(by = 'family_INPADOC_country_count', 
                                ascending = 0,
                                axis = 0)
        
        top_3 = temp.groupby('Topic', group_keys=False).apply(lambda x: x.sort_values('citation_forward_domestic', ascending=False).head(3))
        result = top_3[['Topic','id_wisdomain']]
        
        # Initialize an empty dictionary
        topic_dict = {}
    
        for _, row  in result.iterrows():
            topic, text = row['Topic'], row['id_wisdomain']
            
            if topic not in topic_dict:
                topic_dict[topic] = []
            topic_dict[topic].append(text)
        
        result_dynamic[idx]['Representative_Docs'] = topic_dict.values()
        
        
    #%% 주제 별 레이블링 topic-labelling
    
    for idx in range(len(docs_time_series)) : 
        
        result_topic_info = result_dynamic[idx]
        
        for topic in range(len(result_topic_info)) : 
            print(topic)
            documents = result_topic_info['Representative_Docs'][topic]
            keywords = result_topic_info['Representation'][topic]
        
#             prompt = f"""I have topic described by the following keywords: {keywords}
#             The topic is that contains the following documents: {documents}\n
# Based on the above information, can you give a label of the topic?""".format(documents, keywords)

            instruction = "just return a result without explaination"
            
            prompt = f"""I have topic described by the following keywords: {keywords}
Based on the above information, can you give a korean label of the technology?""".format(keywords)
            
            completion = client.chat.completions.create(
                
            model="gpt-4o",
            # model="gpt-3.5-turbo-0125",
            messages=[
            {"role": "user", 
             "content": prompt},
            {"role" : "system", "content" : instruction}
            ],temperature = 0.0)
            
            label = completion.choices[0].message.content.replace('"', '')
            result_topic_info['Name'][topic] = label
            
        result_dynamic[idx] = result_topic_info
        
        
    #%% 주제별 상위 주제와의 연결
    # trans_dict = {"HBM_일반": "HBM_기타", 
    #               "NPU_일반" : "NPU_기타", 
    #               "NPU_" : "", }
    
    for idx in range(len(docs_time_series)) : 
        
        result_topic_info = result_dynamic[idx]
        docs_df = docs_time_series_cumulative[idx]
        docs_df['domain'] = docs_df['domain'].apply(lambda x : x.split("_")[0]+"_"+x.split("_")[1])
        
        temp = docs_df.groupby(['Topic',"domain"]).size().unstack(fill_value=0)
        
        result_topic_info = pd.concat([result_topic_info, temp], axis = 1)
        result_dynamic[idx] = result_topic_info 
        # for topic in range(len(result_topic_info)) : 
            # docs_df_topic = docs_df.loc[docs_df['Topic'] == topic,:]
    #%%
    
    
    
    #%% 한국/중국 특허 할당
    data_korean_old = copy(data_korean)
    
    abstract_dict = dict(zip(data_korean_old['id_wisdomain'], 
                             data_korean_old['abstract_translated']))
    
    #%%
    # 한국특허 로드
    
    # 위즈도메인 csv 파일이 존재하는 디렉토리 설정
    directory = "D:/OneDrive/특허유니버시아드/데이터/"
    
    file_list = os.listdir(directory)
    file_list = [i for i in file_list if ".csv" in i ]
    
    data_korean = data.loc[data['country'] == 'KR', : ].reset_index(drop = 1)
    
    # gpt : 한글 번역
    
    data_korean['abstract_translated'] = ""
    data_korean['abstract_translated'] = data_korean.apply(lambda x : abstract_dict[x.id_wisdomain] if x.id_wisdomain in abstract_dict.keys() else "", axis = 1)
    
    for idx,row in data_korean.iloc[:,:].iterrows() : 
        
        if row['abstract_translated'] == ""  : 
            
            print(idx)
            abstract = row['abstract']
            
            prompt = f"""Translate to english: {abstract}"""
                
            completion = client.chat.completions.create(
         
            model="gpt-4o",
            # model="gpt-3.5-turbo-0125",
            messages=[
            {"role": "user", 
             "content": prompt},
            {"role" : "system", "content" : instruction}
            ],temperature = 0.0)
            
            abstract_translated = completion.choices[0].message.content.replace('"', '')
            data_korean['abstract_translated'][idx] = abstract_translated
            
            
    #%% 한국어 할당
    docs = data_korean['abstract_translated']
    embeddings_kt = embedding_model.encode(docs, device = "cuda")

    embeddings_kt = umap_model.transform(embeddings_kt) 
    data_korean['Topic'] = 0
    
    for idx, row in data_korean.iterrows(): 
        
        array = embeddings_kt[idx]
        array_dict = {i: float(array[i]) for i in range(len(array))}
    
        cluster= topic_model.hdbscan_model.model.predict_one(x = array_dict)
        data_korean['Topic'][idx]= cluster
        
    
    #%% 중국어 할당
    
    data_chinese = data.loc[data['country'] == 'CN', : ].reset_index(drop = 1)
    
    docs = data_chinese['abstract_']
    
    embeddings_cn = embedding_model.encode(docs, device = "cuda")

    embeddings_cn = umap_model.transform(embeddings_cn) 
    
    data_chinese['Topic'] = 0
    
    for idx, row in data_chinese.iterrows(): 
        print(idx)
        
        array = embeddings_cn[idx]
        array_dict = {i: float(array[i]) for i in range(len(array))}
    
        cluster= topic_model.hdbscan_model.model.predict_one(x = array_dict)
        data_chinese['Topic'][idx]= cluster
    
    #%% 미국특허 할당
    
    data_us = data.loc[data['country'] == 'US', : ].reset_index(drop = 1)
    
    docs = data_us['abstract']
    
    embeddings_us = embedding_model.encode(docs, device = "cuda")

    embeddings_us = umap_model.transform(embeddings_us) 
    
    data_us['Topic'] = -1
    
    for idx, row in data_us.iterrows(): 
        print(idx)
        
        array = embeddings_us[idx]
        array_dict = {i: float(array[i]) for i in range(len(array))}
    
        cluster= topic_model.hdbscan_model.model.predict_one(x = array_dict)
        data_us['Topic'][idx]= cluster
    
    
    # %%
    
    c = Counter(data_us['Topic'])
    
    
    #%%
    
    # result_us = docs_time_series_cumulative[-1]
    
    topic_dict = dict(zip(result_dynamic[2]['Topic'], result_dynamic[2]['Name']))
    
    
    #%% 결과 저장 
    
    data_us['hierarchy_0'] = data_us['domain'].apply(lambda x : x.split("_")[0])
    data_us['hierarchy_1'] = data_us['domain'].apply(lambda x : x.split("_")[1])
    data_us['hierarchy_2'] = data_us['Topic'].apply(lambda x : topic_dict[x])
    
    data_us.to_csv(directory + "output/us_result.csv", index = 0)
    
    data_chinese['hierarchy_0'] = data_chinese['domain'].apply(lambda x : x.split("_")[0])
    data_chinese['hierarchy_1'] = data_chinese['domain'].apply(lambda x : x.split("_")[1])
    data_chinese['hierarchy_2'] = data_chinese['Topic'].apply(lambda x : topic_dict[x])
    
    data_chinese.to_csv(directory + "output/cn_result.csv", index = 0)
    
    data_korean['hierarchy_0'] = data_korean['domain'].apply(lambda x : x.split("_")[0])
    data_korean['hierarchy_1'] = data_korean['domain'].apply(lambda x : x.split("_")[1])
    data_korean['hierarchy_2'] = data_korean['Topic'].apply(lambda x : topic_dict[x])
    
    data_korean.to_csv(directory + "output/kr_result.csv", index = 0)
    
    #%% 기술 통계량
    # result_df = data_input.groupby(['applicant_rep_icon','Topic']).size().unstack(fill_value=0)
    # data_us.groupby(['hierarchy_0']).size().unstack(fill_value = 0)
    data_us['period'] = data_us['year_application'].apply(lambda x : 0 if x < 2011 else (1 if x < 2017 else 2) )
    # data_input = data_us.loc[data_us]
    result = data_us.groupby(['Topic']).size()
    result = result.reset_index(drop = 0)
    
    #%% 주요 출원인 분석
        
    # x축 출원인
    # y축 기술
    topic_main_list = [10,6,0,2,21]
    topic_main_label = [result_topic_info['Name'][i] for i in topic_main_list]
    
    topic_emerging_list = [14,12,13,4,9]
    topic_emerging_label = [result_topic_info['Name'][i] for i in topic_emerging_list]
    topic_emerging_label = [i.replace("컴퓨텍스 인터페이스 기술", "CXL기반 메모리 대역폭 확장 기술") for i in topic_emerging_label]
    topic_emerging_label = [i.replace("신경처리장치 (NPU)", "신경망 처리 및 인공지능 하드웨어 기술") for i in topic_emerging_label]
    
    # topic_list = [i for i in range(0,33)]
    # drop_list = []
    
    
    #%% 시각화 : visualization 기업 포트폴리오 1
    from copy import copy 
    
    data_input = copy(data_us) 
    data_input = data_input.loc[pd.isnull(data_input['applicant_rep_icon']) == False , :].reset_index(drop= 1)
    
    c = Counter(data_input['applicant_rep_icon'])
    
    data_input['applicant_rep_icon'] = data_input['applicant_rep_icon'].apply(lambda x : re.sub('GLOBALFOUNDRIES SINGAPORE PTE', 'GLOBALFOUNDRIES', x))
    data_input['applicant_rep_icon'] = data_input['applicant_rep_icon'].apply(lambda x : re.sub('SK HYNIX', 'HYNIX', x))
    data_input['applicant_rep_icon'] = data_input['applicant_rep_icon'].apply(lambda x : re.sub('TAIWAN SEMICONDUCTOR MANUFACTURING', 'TSMC', x))
    data_input['applicant_rep_icon'] = data_input['applicant_rep_icon'].apply(lambda x : re.sub('INTERNATIONAL BUSINESS MACHINES', 'IBM', x))
    data_input['applicant_rep_icon'] = data_input['applicant_rep_icon'].apply(lambda x : re.sub('STATS CHIPPAC', 'JCET', x))
    data_input['applicant_rep_icon'] = data_input['applicant_rep_icon'].apply(lambda x : re.sub('UNITED MICROELECTRONICS', 'UMC', x))
    data_input['applicant_rep_icon'] = data_input['applicant_rep_icon'].apply(lambda x :x.split(' ')[0])
    
    c = Counter(data_input['applicant_rep_icon'])
    
    temp = c.most_common(10)

    temp=[i[0] for i in temp]
    
    topic_list = [i for i in range(31)]
    
    # topic_list = topic_emerging_list
    # topic_label = topic_emerging_label
    
    data_input = data_input.loc[data_input['Topic'].isin(topic_list), : ]
    data_input = data_input.loc[data_input['applicant_rep_icon'].isin(temp), : ]
    data_input = data_input.loc[data_input['period'] <= 1 , : ]
    
    # List of names in the desired order
    
    result_df = data_input.groupby(['applicant_rep_icon','Topic']).size().unstack(fill_value=0)
    
    if "JCET" not in result_df.index :
        new_row = pd.DataFrame(dict(zip(result_df.columns, [0,0,0,0,0])), index = ['JCET'])
        result_df = pd.concat([result_df, new_row])
    
    desired_order = temp
    
    # Convert 'name' column to a categorical type with the specified order
    result_df.index = pd.Categorical(result_df.index, categories=desired_order, ordered=True)
    
    # Sort the DataFrame by the 'name' column
    result_df = result_df.sort_index()

    # result_df = result_df[topic_list]
    
    # result_df.columns = topic_label #라벨 여부 결정
    
    # result_df['columns'] = 
    import seaborn as sns
    import matplotlib.pyplot as plt
    plt.rcParams['font.family'] ='Malgun Gothic'
    
    
    plt.figure(figsize=(15,10))
    
    ax = sns.heatmap(result_df, annot=True, linewidths=.5, 
                      square = True, 
                       cmap = 'Greens', fmt='g', cbar=False , annot_kws = {"size" : 18})
                      # cmap = 'Blues', fmt='g', cbar=False , annot_kws = {"size" : 18})
                     # cmap = 'Greys', fmt='g', cbar=False , annot_kws = {"size" : 14})
    
    # ax.xaxis.tick_top()    
    ax.tick_params(labelsize=16)
    
    # Rotate the x-axis labels
    ax.set_xticklabels(ax.get_xticklabels(), rotation=15, horizontalalignment='center')
    ax.xaxis.tick_top()
    
    plt.savefig(directory+'output/main_emerging.png',dpi= 1000, bbox_inches='tight',)
    plt.show()
    
    #%% 국가별 포트폴리오
    from copy import copy 
    
    data_input = copy(data_us) 
    data_input = data_input.loc[pd.isnull(data_input['applicant_country']) == False , :].reset_index(drop= 1)
    
    data_input['applicant_country'] = data_input['applicant_country'].apply(lambda x : x[0])
    
    data_input = data_input.loc[data_input["period"] < 2 ,:]

    
    c = Counter(data_input['applicant_country'])
    
    temp = c.most_common(7)


    temp=[i[0] for i in temp]
    
    # topic_list = [i for i in range(35)]
    # topic_list = [i for i in topic_list if i not in [18,20,22]]
    # topic_list = [0,3,5,6,10,17,19,21,23,28,31,32] # HBM
    # topic_list = [1,2,4,7,8,11,13,14,15,16,26,27,29,34] # 뉴로모픽
    topic_list = [12,9,24,25,30] # 기타
    
    # topic_label = topic_emerging_label
    
    data_input = data_input.loc[data_input['Topic'].isin(topic_list), : ]
    data_input = data_input.loc[data_input['applicant_country'].isin(temp), : ]
    
    
    # List of names in the desired order
    
    result_df = data_input.groupby(['applicant_country','Topic']).size().unstack(fill_value=0)
    
    desired_order = temp
    
    # Convert 'name' column to a categorical type with the specified order
    result_df.index = pd.Categorical(result_df.index, categories=desired_order, ordered=True)
    
    # Sort the DataFrame by the 'name' column
    result_df = result_df.sort_index()
    result_df = result_df[topic_list]
    # result_df.columns = topic_label
    
   
    
    # result_df['columns'] = 
    import seaborn as sns
    import matplotlib.pyplot as plt
    plt.rcParams['font.family'] ='Malgun Gothic'
    
    plt.figure(figsize=(10,10))
    
    ax = sns.heatmap(result_df, annot=True, linewidths=.5, 
                      square = True, 
                        # cmap = 'Reds', fmt='g', cbar=False , annot_kws = {"size" : 18})
                       # cmap = 'Blues', fmt='g', cbar=False , annot_kws = {"size" : 18})
                       cmap = 'Greys', fmt='g', cbar=False , annot_kws = {"size" : 24})
    
    
    # ax.xaxis.tick_top()    
    # ax.tick_params(labelsize=18)
    ax.tick_params(labelsize=24)
    
    # Rotate the x-axis labels
    # ax.set_xticklabels(ax.get_xticklabels(), rotation=15, horizontalalignment='center')
    ax.xaxis.tick_top()
    
    plt.savefig(directory+'output/country.png',dpi= 1000, bbox_inches='tight',)
    plt.show()
    
    
    
    
    #%% 한,미,중 top-10
    
    from collections import Counter
    
    import plotly.express as px
    import plotly.io as pio
    
    pio.renderers.default='browser'
    
    data_input = copy(data_us) 
    data_input = data_input.loc[pd.isnull(data_input['applicant_rep_icon']) == False , :].reset_index(drop= 1)
    
    c = Counter(data_input['applicant_rep_icon'])
    # c = Counter(data_input['applicant_rep'])
    
    data_input['applicant_rep_icon'] = data_input['applicant_rep_icon'].apply(lambda x : re.sub('GLOBALFOUNDRIES SINGAPORE PTE', 'GLOBALFOUNDRIES', x))
    data_input['applicant_rep_icon'] = data_input['applicant_rep_icon'].apply(lambda x : re.sub('SK HYNIX', 'HYNIX', x))
    data_input['applicant_rep_icon'] = data_input['applicant_rep_icon'].apply(lambda x : re.sub('TAIWAN SEMICONDUCTOR MANUFACTURING', 'TSMC', x))
    data_input['applicant_rep_icon'] = data_input['applicant_rep_icon'].apply(lambda x : re.sub('INTERNATIONAL BUSINESS MACHINES', 'IBM', x))
    data_input['applicant_rep_icon'] = data_input['applicant_rep_icon'].apply(lambda x : re.sub('STATS CHIPPAC', 'JCET', x))
    data_input['applicant_rep_icon'] = data_input['applicant_rep_icon'].apply(lambda x : re.sub('UNITED MICROELECTRONICS', 'UMC', x))
    data_input['applicant_rep_icon'] = data_input['applicant_rep_icon'].apply(lambda x :x.split(' ')[0])
    
    c = Counter(data_input['applicant_rep_icon'])
    
    temp = c.most_common(30)

    temp=[i[0] for i in temp]
    
    # topic_list = topic_emerging_list
    # topic_label = topic_emerging_label
    
    # data_input = data_input.loc[data_input['Topic'].isin(topic_list), : ]
    data_input = data_input.loc[data_input['applicant_rep_icon'].isin(temp), : ]
    data_input = data_input.loc[data_input['period'] == 1, : ]
    # data_input['hierarchy_1'] = data_input.apply(lambda x : x.hierarchy_0+ "_" + x.hierarchy_1, axis =1)
    
    df = data_input.groupby(['applicant_rep_icon','hierarchy_0','hierarchy_1']).size()
    
    df = df.reset_index(drop = 0)
    
    
    fig = px.sunburst(df, path=['hierarchy_0','hierarchy_1','applicant_rep_icon',], values= 0,
                      labels= 0,
                      maxdepth = 3,
                      color='hierarchy_0', 
                      color_discrete_map={'(?)':'black', 'HBM':'blue', '뉴로모픽':'red', 'NPU' : 'green', "PIM" : "purple"}
                      # hover_data=['iso_alpha'],
                      # color_continuous_scale='RdBu',
                      # color_continuous_midpoint=np.average(df['lifeExp'], weights=df['frequency']))
                      )
    
    fig.update_layout(font=dict(size=48))
    fig.update_traces(textinfo="label+percent parent")
    fig.update_layout(margin=dict(t=0, b=0, l=0, r=0))                  
    fig.show()
    # fig.savefig(directory+'output/2017.png',dpi= 1000, bbox_inches='tight',)
    # fig.to_image()
    
    
    
    #%%
    c = Counter(x for xs in data_input['CPC_sg'] for x in set(xs))
    threshold = 3
    c = {key: val for key, val in c.items() if val >= threshold}
    
    df = pd.DataFrame(c.items())
    
    df.columns = ['subgroup', 'frequency']
    df['maingroup'] = df['subgroup'].apply(lambda x : x.split('/')[0])
    df['subclass'] = df['maingroup'].apply(lambda x : x[0:4])
    df['mainclass'] = df['subclass'].apply(lambda x : x[0:3])
    
    fig = px.sunburst(df, path=['subclass', 'maingroup'], values='frequency',
                      labels= 'frequency',
                      maxdepth = 2,
                       color='mainclass', 
                       # hover_data=['iso_alpha'],
                      color_continuous_scale='RdBu',
                      # color_continuous_midpoint=np.average(df['lifeExp'], weights=df['frequency']))
                      )
    
    fig.update_layout(
    
    font=dict(size=48))
    fig.update_layout(margin=dict(t=0, b=0, l=0, r=0))                  
    fig.show()
    
    #%% 점유?
    
    data_input = copy(data_us) 
    
    def calculate_hhi(market_shares):
        """
        Calculate the Herfindahl-Hirschman Index (HHI).
    
        Parameters:
        market_shares (list or np.array): List or array of market shares (in percentage, e.g., [30, 20, 20, 10, 10, 10])
    
        Returns:
        float: HHI value
        """
        # Convert market shares to proportions
        market_shares = np.array(market_shares) / 100
        
        # Calculate HHI
        hhi = np.sum(market_shares**2) * 10000  # multiplying by 10000 to get the index in percentage points
        
        return hhi
    
    
    result = data_input.groupby(['Topic'])['applicant_rep_icon'].value_counts(normalize = 1).mul(100)
    temp = result.groupby(["Topic"]).agg(lambda x : calculate_hhi(x))
    
    temp = temp.reset_index(drop = 0)
    # result = result.reset_index(drop = 0)
    # calculate_hhi()
    
    #%%
    temp = data_input.groupby(['Topic','applicant_rep_icon']).size()
    temp = temp.reset_index(drop = 0)
    #%%
    temp = temp.sort_values(by = 0, 
                            ascending = 0,
                            axis = 0)
    
    #%%
    
    top_3 = temp.groupby('Topic', group_keys=False).apply(lambda x: x.sort_values(0, ascending=False).head(1))
    
    #%% 
    temp = data_chinese.groupby(['hierarchy_0','hierarchy_1']).count()

    #%%
    result = top_3[['Topic','id_wisdomain']]
    
    #%% garbage
    
    directory = "D:/OneDrive/특허유니버시아드/데이터/"
    
    file_list = os.listdir(directory)
    file_list = [i for i in file_list if ".csv" in i ]
    
    data_chinese = pd.DataFrame()
    domain_dict = {}
    
    for file in file_list : 
        
        data_temp = pd.read_csv(directory + file, skiprows= 4)
        
        data_temp = preprocess.wisdomain_prep(data_temp)   
        data_temp['country'] = data_temp['id_wisdomain'].apply(lambda x : x[0:2])
        data_temp = data_temp.loc[data_temp['country'] == 'CN', :]
        
        data_chinese = pd.concat([data_chinese, data_temp], axis = 0).reset_index(drop = 1)
        
    
    data_chinese = data_chinese.drop_duplicates(subset= ['id_wisdomain']).reset_index(drop = 1)
    
    #%%
    
    
    
    
#%% field classify
instruction = "Extract the actual text instead of generating it."
data_us["field_preprocessed"] = ""

for idx, row in data_us.iterrows() : 
    if data_us['field_preprocessed'][idx] == "" :
        print(idx)
        
        prompt = "Extract the parent technical field and child technical field from the sentence.: \n"
        prompt = "Q : "
        prompt += "The present invention relates to piezoelectric materials and, more particularly, to the integration of piezoelectric materials with substrates."
        prompt += "\nA : "
        prompt += '{"Parent" : "piezoelectric materials", "Child" : "the integration of piezoelectric materials with substrates"}'
        prompt += "\nQ : "    
        prompt += row['field_generated']
        prompt += "\nA : "    
        # print(prompt)
        
        completion = client.chat.completions.create(
            model="gpt-4o",
            # model="gpt-3.5-turbo-0125",
            messages=[
            {"role": "user", 
             "content": prompt},
            {"role" : "system", "content" : instruction}
            ],temperature = 0.1)
        
        data_us["field_preprocessed"][idx] = completion.choices[0].message.content 
    
# print(prompt)
# print(completion.choices[0].message.content )

#%%
import ast 

def safe_literal_eval(s):
    try:
        return ast.literal_eval(s)
    except (ValueError, SyntaxError) as e:
        return f"Error: {e}"


#%%
data_us['field_parsed'] = data_us['field_preprocessed'].apply(lambda x : 
                                                        safe_literal_eval(x))

temp = data_us['field_parsed']
# temp = temp[3]
# ast.literal_eval(temp)
# ast.literal_eval(data['field_preprocessed'][0])

#%%  field re-scraping
error = "Error: invalid syntax (<unknown>, line 1)"


instruction = "Extract the actual text instead of generating it."

for idx, row in data.iterrows() : 
    if row['field_parsed'] == error :
        print(idx)
        
        prompt = "Extract the parent technical field and child technical field from the sentence.: \n"
        prompt = "Q : "
        prompt += "The present invention relates to piezoelectric materials and, more particularly, to the integration of piezoelectric materials with substrates."
        prompt += "\nA : "
        prompt += '{"Parent" : "piezoelectric materials", "Child" : "the integration of piezoelectric materials with substrates"}'
        prompt += "\nQ : "    
        prompt += row['field_generated']
        prompt += "\nA : "    
        # print(prompt)
        
        completion = client.chat.completions.create(
            model="gpt-4o",
            # model="gpt-3.5-turbo-0125",
            messages=[
            {"role": "user", 
             "content": prompt},
            {"role" : "system", "content" : instruction}
            ],temperature = 0.1)
        
        data["field_preprocessed"][idx] = completion.choices[0].message.content 
   
    docs_input = pd.DataFrame()
    
    for idx,row in data.iterrows() : 
        
        if type(row['field_parsed']) != dict :continue
        
        Parent = row['field_parsed']['Parent']
        Child = row['field_parsed']['Child']
        pt_id = row['id_wisdomain']
        year = row['year_application']
        applicant = row['applicant_rep']
        if type(Parent) == list : continue
        if type(Child) == list : continue
        
        temp = pd.DataFrame({'pt_id' : [pt_id],
                             'year' : [year],
                             'applicant' : [applicant],
                             'document' : [Parent],
                             'type' : ["Parent"]})
        
        docs_input = pd.concat([docs_input, temp], axis = 0 )
        
        temp = pd.DataFrame({'pt_id' : [pt_id],
                             'year' : [year],
                             'applicant' : [applicant],
                             'document' : [Child],
                             'type' : ["Child"]})
        
        docs_input = pd.concat([docs_input, temp], axis = 0 )
        
    docs_input = docs_input.reset_index(drop = 1)
    