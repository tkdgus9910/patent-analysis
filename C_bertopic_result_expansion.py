# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 14:25:34 2024

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
        
    # 위즈도메인 csv 파일이 존재하는 디렉토리 설정
    directory_ = 'D:/OneDrive/연구실 과제(현대차_진섭)/2. 기능분석/특허 데이터/Raw/'
    
    directory = directory.replace("\\", "/") # window
    sys.path.append(directory+'/submodule')
    import preprocess
    
    directory_ = 'D:/OneDrive/연구실 과제(현대차_진섭)/2. 기능분석/2차워크숍 기초자료/'
    
    with open(directory_+ 'claims_decomposed.pkl', 'rb') as f:
        data_input = pickle.load(f)
        
    
    data_input = data_input.loc[data_input['claims_decomposed'].apply(lambda x : len(x) > 0 ),:].reset_index(drop = 1)
    import ast
    import re    
    
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
     
    #%% 필터링 load
    
    directory = 'D:/OneDrive/연구실 과제(현대차_진섭)/2. 기능분석/2차워크숍 기초자료/level3/'
    
    trans_dict_mech = {}
    
    temp = pd.read_excel(directory + 'translation_mech.xlsx')
    val = 'nan'
    for idx, row in temp.iterrows() : 
        key = row['3차']
        if str(row['2차']) != 'nan' :  #갱신
            val = str(row['2차'])
            
        trans_dict_mech[key] = val
        
    #%% 
    temp = pd.read_excel(directory + 'translation_elec.xlsx')
    
    trans_dict_elec = {}
    
    val = 'nan'
    for idx, row in temp.iterrows() : 
        key = row['3차']
        if str(row['2차']) != 'nan' :  #갱신
            val = str(row['2차'])
            
        trans_dict_elec[key] = val
    
    
    
    #%% 2404021 도메인과 연결
    
    directory = 'D:/OneDrive/연구실 과제(현대차_진섭)/2. 기능분석/2차워크숍 기초자료/분야 클러스터링/'
    data_domain = pd.read_excel(directory + 'reference_fields_document.xlsx')
    
    domain_dict = {}
    domain_dict[1] = 'Valve pump inlet mechanism'
    domain_dict[2] = 'Electrical connector'
    domain_dict[3] = 'Transmission mechanism'
    domain_dict[8] = 'Door lock mechanism'
    domain_dict[9] = 'Seat assembly'
    domain_dict[12] = 'Wheel mechanism'
    domain_dict[13] = 'Sensor device'
    domain_dict[18] = 'Electronic device'
    domain_dict[21] = 'Brake mechanism'
    domain_dict[24] = 'Robot arm'
    
    keys = list(domain_dict.keys())
    # keys = [str(i) for i in keys]
    data_domain = data_domain.loc[data_domain['Topic'].apply(lambda x : True if x in keys else False),:]
    
    data_input['key'] = data_input['claims'] + data_input['title'] + data_input['abstract']
    # data_input['key'] = data_input['key'].apply(lambda x : re.sub('\ufeff', '' ,x).strip())
    
    data_input['domain'] = ''
    
    for idx, row in data_domain.iterrows() : 
        key = row['Document'].strip()
        try : 
            idx_ = list(data_input.loc[data_input['key'] == key, 'id_wisdomain'].index)[0]
            data_input['domain'][idx_] = domain_dict[row['Topic']]
        except :
            key = re.sub('\ufeff', '' ,key).strip()
            try : 
                idx_ = list(data_input.loc[data_input['key'] == key, 'id_wisdomain'].index)[0]
                data_input['domain'][idx_] = domain_dict[row['Topic']]
            except :
                pass
       

    #%% bertopic 결과 load
    
    directory_docs = 'D:/OneDrive/연구실 과제(현대차_진섭)/2. 기능분석/2차워크숍 기초자료/level3/docs/'
    directory_topics = 'D:/OneDrive/연구실 과제(현대차_진섭)/2. 기능분석/2차워크숍 기초자료/level3/topics/'
    file_ids = os.listdir(directory_docs)
    file_ids = [i for i in file_ids if '.xlsx' in i ]
    file_ids = [i.split('_')[0]+'_'+i.split('_')[1] for i in file_ids]
    
    
    #%% 240402 대표특허 요약문 수정
    
    for file_id in file_ids : 
        
        directory_topics_output = 'D:/OneDrive/연구실 과제(현대차_진섭)/2. 기능분석/2차워크숍 기초자료/level3/topics/output/'
        file_list_output = os.listdir(directory_topics_output)
        output = file_id + '_docs.xlsx'
        
        if output not in file_list_output :
            print(file_id)
            
            # read docs
            directory_file = directory_docs + file_id + '_docs.xlsx'
            data_docs = pd.read_excel(directory_file, index_col= 0)
            
            # read topics
            directory_file = directory_topics + file_id + '_topics.xlsx'
            data_topics = pd.read_excel(directory_file, index_col= 0,)
            columns = data_topics.columns
            columns = [i for i in columns if 'Unnamed' not in i]
            data_topics = data_topics[columns]
            
            data_topics['요약문'] = '' # LLM
            data_topics['대표 특허 명칭'] = '' # 가능
            data_topics['대표 CPC'] = '' # 가능
            data_topics['자주 활용되는 분야'] = '' # 외부 연결
            
            data_topics['평균 피인용수'] = ''
            data_topics['평균 출원연도'] = ''
            
            for col_name in  ['CPC_sc', 'CPC_mg', 'CPC_sg', '핵심기술', '주요출원인'] : 
                for text in  ['_top1', '_top2', '_top3'] :
                    data_topics[col_name+text] = ''
                    data_topics[col_name+text+'_count'] = ''
            
            for topic in data_topics.index : 
                if '최종 label' in data_topics.columns : 
                    if data_topics['최종 label'][topic] not in list(trans_dict.keys()) :
                        # print(data_topics['label'][idx] + ' filtered')
                        continue
                else : 
                    if data_topics['label'][topic] not in list(trans_dict.keys()) :
                        # print(data_topics['label'][idx] + ' filtered')
                        continue
                    
                # id_wisdomain
                patent_id_list = list(set(data_docs.loc[data_docs['Topic'] == topic, 'id_wisdomain']))
                
                # 데이터 연결 1
                data_temp = data_input.loc[data_input['id_wisdomain'].apply(lambda x : True if x in patent_id_list else False), : ].reset_index(drop = 1)
                
                # 대표 특허 명칭
                temp = data_docs.loc[data_docs['Topic'] == topic].reset_index(drop = 1)
                temp = temp.loc[temp['Representative_document'] == 1, :].reset_index(drop = 1)
                temp = temp.sort_values('similarity_2nd')['id_wisdomain']
                
                titles = list(data_input.loc[data_input['id_wisdomain'].apply(lambda x : True if x in list(temp) else False),'title'])
            
                data_topics['대표 특허 명칭'][topic] = set(titles)
                
                # 대표 CPC 리스트
                c = Counter(x for xs in data_temp['CPC_sg'] for x in set(xs))
                c = c.most_common(3)
                c = [i[0] for i in c]
                data_topics['대표 CPC'][topic] = c
                
                # 요약문 
                temp = data_docs.loc[data_docs['Topic'] == topic].reset_index(drop = 1)
                temp = temp.sort_values('similarity_2nd', ascending = 0)[0:20]
                text = "\n".join(temp['Document'])
                
                instruction= """Provide results simply without explanation.
* Present the resulting sentence with a newline like
- A sentence
- B sentence
- C sentence"""
                if file_id.split('_')[1] == 'mechanical' :
                    first_level = 'mechanical attachment and detachment'
                else : 
                    first_level = 'electrical attachment and detachment'
                    
                prompt = """Using the information in the given sentences, explain the '{}' technology from the perspective of '{} mechanism' in three sentences.""".format(data_topics['label'][topic], first_level)
                prompt += '\n['
                prompt += text
                prompt += ']'
                
                completion = client.chat.completions.create(
                    model="gpt-4-0125-preview",
                    # model="gpt-3.5-turbo-0125",
                    messages=[
                    {"role": "user", 
                      "content": prompt},{"role" : "system", "content" : instruction}
                    ])
                
                data_topics['요약문'][topic] = completion.choices[0].message.content
                
                # 분야
                c = Counter(data_temp['domain'])
                c.pop('')
                c = c.most_common(1)
                try : 
                    data_topics['자주 활용되는 분야'][topic] = c[0][0]
                except : pass
                
                # 부가 정보: 주요 CPC
                c = Counter(x for xs in data_temp['CPC_sc'] for x in set(xs))
                c = c.most_common(3)
                for rank, text in enumerate(['_top3', '_top2','_top1']) :
                    item = c.pop()
                    data_topics['CPC_sc' + text][topic] = item[0]
                    data_topics['CPC_sc' + text+'_count'][topic] = item[1]
                
                c = Counter(x for xs in data_temp['CPC_mg'] for x in set(xs))
                c = c.most_common(3)
                for rank, text in enumerate(['_top3', '_top2','_top1']) :
                    item = c.pop()
                    data_topics['CPC_mg' + text][topic] = item[0]
                    data_topics['CPC_mg' + text+'_count'][topic] = item[1]
                    
                c = Counter(x for xs in data_temp['CPC_sg'] for x in set(xs))
                c = c.most_common(3)
                for rank, text in enumerate(['_top3', '_top2','_top1']) :
                    item = c.pop()
                    data_topics['CPC_sg' + text][topic] = item[0]
                    data_topics['CPC_sg' + text+'_count'][topic] = item[1]
                    
                data_temp['citation_forward_domestic_count'] = data_temp['citation_forward_domestic_count'].fillna(0) 
                
                # 핵심 기술
                # 피인용수 높은 기술 top 3
                data_temp = data_temp.sort_values('citation_forward_domestic_count', ascending = 0 )
                temp = dict(zip(data_temp['id_wisdomain'], data_temp['citation_forward_domestic_count']))
                temp = list(temp.items())[:3]
                for rank, text in enumerate(['_top3', '_top2','_top1']) :
                    item = temp.pop()
                    if item[1] == 0 : continue
                    data_topics['핵심기술' + text][topic] = item[0]
                    data_topics['핵심기술' + text+'_count'][topic] = item[1]
                    
                # 주요 출원인 top 3
                c = Counter(data_temp['applicant_rep_icon'])
                try : c.pop('UNASSIGNED')
                except : pass
                c = c.most_common(3)
                
                for rank, text in enumerate(['_top3', '_top2','_top1']) :
                    if len(c) > 0 :
                        item = c.pop()
                        if item[1] == 0 : continue
                        if str(item[0]) == 'nan' : continue
                        data_topics['주요출원인' + text][topic] = item[0]
                        data_topics['주요출원인' + text+'_count'][topic] = item[1]
                    
                data_temp['citation_forward_domestic_count'] = data_temp['citation_forward_domestic_count'].fillna(0) 
                
                # 평균 피인용수 
                data_topics['평균 피인용수'][topic] = np.mean(data_temp['citation_forward_domestic_count'])
                
                # 평균 출원연도
                data_topics['평균 출원연도'][topic] = np.mean(data_temp['date_application_strp']).year
                
                
            # 결과 저장
            directory_topics_output = 'D:/OneDrive/연구실 과제(현대차_진섭)/2. 기능분석/2차워크숍 기초자료/level3/topics/output/'
            data_topics = data_topics.loc[data_topics['대표 특허 명칭'].apply(lambda x : True if len(x) != 0  else False) , :].reset_index(drop = 1)
            data_topics.to_excel(directory_topics_output+ file_id + '_docs.xlsx')
                
            
        
    #%% 240402 대표특허 수정
    
    file_list_output = os.listdir(directory_topics_output)
    file_list_output = [i for i in file_list_output if '.xlsx' in i ]
    
    for file in file_list_output:
        print(file)
        data_topics = pd.read_excel(directory_topics_output+file, index_col=0)
        file_name = file.split('_')[0] +'_' +file.split('_')[1] + '_topics_filtered.xlsx'
        data_topics.to_excel(directory_topics_output+ file_name)
    
    #%%
        # read docs
        file_id = file.split('_')[0] + '_' +file.split('_')[1]
        directory_file = directory_docs + file_id + '_docs.xlsx'
        data_docs = pd.read_excel(directory_file, index_col= 0)
       
        # id_wisdomain
        patent_id_list = list(set(data_docs.loc[data_docs['Topic'] == file, 'id_wisdomain']))
        
        # 데이터 연결 1
        data_temp = data_input.loc[data_input['id_wisdomain'].apply(lambda x : True if x in patent_id_list else False), : ].reset_index(drop = 1)
        
        for idx, topic in enumerate(data_topics['Topic']) : 
            # 대표 특허 명칭
            temp = data_docs.loc[data_docs['Topic'] == topic].reset_index(drop = 1)
            temp = temp.loc[temp['Representative_document'] == 1, :].reset_index(drop = 1)
            temp = temp.sort_values('similarity_2nd')['id_wisdomain']
            
            titles = list(data_input.loc[data_input['id_wisdomain'].apply(lambda x : True if x in list(temp) else False),'title'])
        
            data_topics['대표 특허 명칭'][idx] = set(titles)
    
        # 결과 저장
        file = file_id +'_revised.xlsx'
        data_topics.to_excel(directory_topics_output+file)
    
    #%% 240403 클러스터 업보 생성 : 다른 군집에 동일 특허가 존재할 경우
    
    data_input['cluster_stack'] = 0
    wisdomain_id2idx = dict(zip(data_input['id_wisdomain'], data_input.index))
    
    for file_id in file_ids : 
        
        # read docs
        directory_file = directory_docs + file_id + '_docs.xlsx'
        data_docs = pd.read_excel(directory_file, index_col= 0)
        
        patent_id_list = list(set(data_docs.loc[data_docs['Topic'] != -1,'id_wisdomain']))
        
        idcies = [wisdomain_id2idx[i] for i in patent_id_list]
        data_input['cluster_stack'][idcies] += 1
        

    #%% 240403 토픽 최종 필터링
   
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    
    def average_cosine_similarity(doc_embeddings):
        """
        Calculate the average cosine similarity between documents embedded in 768 dimensions.
        
        Parameters:
        - doc_embeddings (np.array): A 2D numpy array where each row represents a document embedding.
        
        Returns:
        - float: The average cosine similarity between all pairs of documents.
        """
        # Ensure the input is a numpy array
        doc_embeddings = np.array(doc_embeddings)
        
        # Check if the embeddings are of the correct shape (N, 768)
        if doc_embeddings.shape[1] != 768:
            raise ValueError("Document embeddings must be in 768 dimensions.")
        
        # Compute the cosine similarity matrix
        sim_matrix = cosine_similarity(doc_embeddings)
        
        # Exclude the diagonal elements and compute the average similarity
        np.fill_diagonal(sim_matrix, 0)
        avg_similarity = np.sum(sim_matrix) / (sim_matrix.shape[0] * (sim_matrix.shape[1] - 1))
        
        return avg_similarity

    #%% 240404 업데이트
    from sentence_transformers import SentenceTransformer
    from scipy import stats
    
    embedding_model = SentenceTransformer('AI-Growth-Lab/PatentSBERTa')
    
    directory = 'D:/OneDrive/연구실 과제(현대차_진섭)/2. 기능분석/2차워크숍 기초자료/level3/'
    
    first_category = 'mechanical' # 1차 위계 설정
    
    trans_dict = {}
    
    if first_category == 'mechanical' : 
        temp = pd.read_excel(directory + 'translation_mech.xlsx')
    else :
        temp = pd.read_excel(directory + 'translation_elec.xlsx')
    val = 'nan'
    for idx, row in temp.iterrows() : 
        key = row['3차']
        if str(row['2차']) != 'nan' :  #갱신
            val = str(row['2차'])
        trans_dict[key] = val
            
    # Inverting the dictionary to group keys with same values
    inverse_dict = {}
    for key, value in trans_dict.items():
        inverse_dict.setdefault(value, []).append(key)

    
    file_ids = os.listdir(directory_docs)
    file_ids = [i for i in file_ids if '.xlsx' in i ]
    file_ids = [i.split('_')[0]+'_'+i.split('_')[1] for i in file_ids]
    
    file_ids = [i for i in file_ids if first_category in i] # 1 mechanical 먼저 
    
    directory_topics_output = 'D:/OneDrive/연구실 과제(현대차_진섭)/2. 기능분석/2차워크숍 기초자료/level3/topics/output/'
    
    data_topics_total = pd.DataFrame()
    
    # engineering_cpc = ['B'+ str(i).zfill(2) for i in range(1,34)]
    # engineering_cpc += ['F'+ str(i).zfill(2) for i in range(1,29)]
    # engineering_cpc += ['E'+ str(i).zfill(2) for i in [5,6]]
    # engineering_cpc += ['G'+ str(i).zfill(2) for i in [1,2,5,6,7,8,11,12,21]]
    # engineering_cpc += ['H'+ str(i).zfill(2) for i in [1,2,3,4,5]]
    # engineering_cpc += ['Y'+ str(i).zfill(2) for i in [10, ]]
    
    for file_id in file_ids : 
    
        print(file_id)
        # read docs
        directory_file = directory_docs + file_id + '_docs.xlsx'
        data_docs = pd.read_excel(directory_file, index_col= 0)
        
        # read topics
        directory_file = directory_topics_output + file_id + '_topics_filtered.xlsx'
        data_topics = pd.read_excel(directory_file, index_col= 0,)
        if len(data_topics) == 0 : continue
        
        keys = inverse_dict[file_id.split('_')[0]]
        
        
        data_topics = data_topics.loc[data_topics['label'].apply(lambda x : True if x in keys else False), :].reset_index(drop= 1)
        
        data_topics['cpc_diversity'] = 0
        data_topics['cluster_another_count'] = 0
        # data_topics['cpc_engineering_ratio'] = 0
        data_topics['coherence'] = 0
        
        # 주제 별
        for idx, topic in enumerate(data_topics['Topic']) : 
            
            # 주제 데이터 로드
            data_topics_sample = data_docs.loc[data_docs['Topic'] == topic].reset_index(drop = 1)
            
            # 원본 데이터 로드
            patent_id_list = list(set(data_docs.loc[data_docs['Topic'] == topic, 'id_wisdomain']))
            data_temp = data_input.loc[data_input['id_wisdomain'].apply(lambda x : True if x in patent_id_list else False), : ].reset_index(drop = 1)
            
            # 5. embedding 후 평균 유사도 산출
            embeddings_temp = embedding_model.encode(data_topics_sample['Document'])
            data_topics['coherence'][idx] = average_cosine_similarity(embeddings_temp)
            
            # 2. cpc 다양성 계산
            c = Counter(x for xs in data_temp['CPC_sc'] for x in set(xs))
            # data_topics['cpc_diversity'][idx] = len(c) / np.sqrt(len(data_temp['CPC_mg'])) # 군집 크기 역보정
            data_topics['cpc_diversity'][idx] = len(c) / np.sqrt(len(data_temp['CPC_sc'])) # 군집 크기 역보정
            
            # 3. 외부 클러스터 포함량 계산
            data_topics['cluster_another_count'][idx] = data_temp['cluster_stack'].mean()-1
            
            # +alpha. engineering cpc 비중 계산
            # c = Counter(x for xs in data_temp['CPC_mc'] for x in set(xs))
            # keys = list(c.keys())
            # keys = [i for i in keys if i in engineering_cpc]
            # sum_of_values = sum(c[key] for key in keys)
            # total = sum(c.values())
            # data_topics['cpc_engineering_ratio'][idx] = sum_of_values / total
            
        data_topics['2nd_cluster'] = file_id
        
        data_topics_total = pd.concat([data_topics_total, data_topics], axis = 0)
     
    
    # get indicator
    
    data_topics_total['Count'] = data_topics_total['Count'].apply(lambda x : int(x))
    data_topics_total['ind_volume'] = stats.zscore(data_topics_total['Count'], nan_policy='omit', ddof=0)
    data_topics_total['ind_cpc_diversity'] = stats.zscore(data_topics_total['cpc_diversity'], nan_policy='omit', ddof=0)
    
    data_topics_total['ind_cluster_another_count'] = stats.zscore(data_topics_total['cluster_another_count'], nan_policy='omit', ddof=0)
    data_topics_total['ind_coherence'] = stats.zscore(data_topics_total['coherence'], nan_policy='omit', ddof=0)
    
    # data_topics_total['ind_total'] = data_topics_total['ind_cpc_diversity'] - data_topics_total['ind_cluster_another_count'] +data_topics_total['ind_coherence'] 
    # data_topics_total['ind_cpc_engineering_ratio'] = stats.zscore(data_topics_total['cpc_engineering_ratio'], nan_policy='omit', ddof=0)
    data_topics_total['ind_total'] = data_topics_total['ind_cpc_diversity'] - data_topics_total['ind_cluster_another_count'] +data_topics_total['ind_coherence'] 
    # data_topics_total['ind_total'] = data_topics_total['ind_cpc_diversity'] - data_topics_total['ind_cluster_another_count'] +data_topics_total['ind_coherence'] +data_topics_total['cpc_engineering_ratio']  
    
    #%% 결과저장
    
    result = {}
    
    for cluster in file_ids :
        data_temp = data_topics_total.loc[data_topics_total['2nd_cluster'] == cluster , :]
        if len(data_temp) == 0 : continue
        data_temp = data_temp[['2nd_cluster',
                               'Topic', 'label', 
                               'Count', 
                               'cpc_diversity',
                               'cluster_another_count',
                               'coherence',
                               'ind_cpc_diversity', 
                               'ind_coherence',
                               'ind_cluster_another_count', 'ind_total']]
        result[cluster] = data_temp
    #%%
            
    # Create a Pandas Excel writer using XlsxWriter as the engine.
    writer = pd.ExcelWriter(directory_topics_output + first_category +'_토픽지표.xlsx', engine='xlsxwriter')
    data_topics_total_ = data_topics_total[['2nd_cluster',
                            'Topic', 'label', 
                           'Count', 
                           'cpc_diversity',
                           'cluster_another_count',
                           'coherence',
                           'ind_cpc_diversity', 
                           'ind_coherence',
                           'ind_cluster_another_count', 'ind_total']]

    #%%
    data_topics_total_.to_excel(writer, sheet_name='total', index=False)
    # Write each dataframe to a different worksheet.
    for sheet_name, df in result.items():
        sheet_name = sheet_name.split('_')[0][:30]
        df.to_excel(writer, sheet_name=sheet_name, index=False)
    
    # Close the Pandas Excel writer and output the Excel file.
    writer.close()
    