# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 17:28:37 2024

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
    directory += 'LLM/'
    
    file_list = os.listdir(directory)
    
    # 데이터 로드 
    with open(directory + 'data_preprocessed.pkl', 'rb') as file:  
        data_preprocessed = pickle.load(file)  
        
    
    #%% 1-1. 전체 시계열 출원 동향 
    
    from copy import copy
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    data_input = copy(data_preprocessed.loc[data_preprocessed['year_application'] <= 2023, :].reset_index(drop= 1))
    data_input = copy(data_input.loc[data_input['year_application'] >= 2001, :].reset_index(drop= 1))
    
    year_counts = data_input['year_application'].value_counts().sort_index()
    
    plt.figure(figsize=(12, 6))
    sns.lineplot(x=year_counts.index, y=year_counts.values, marker='o', color='green', label = 'count')
    
    plt.xlabel('Year',fontsize = 15)
    plt.ylabel('Annual Counts',fontsize = 15)
    plt.title('Patent Application Trend',fontsize = 15)
    
    # x축 ticks 조정
    ticks = list(set(data_input['year_application']))
    ticks = [i for i in ticks if i%3 == 0 ]
    plt.xticks(ticks)  # Rotate x-axis labels
    plt.grid(axis='y')
    
    plt.tight_layout()  # Ensure everything fits nicely
    
    # The arrow starts at (2,10) and ends at (8,10)
    plt.axvline(x = 2022, color = 'r', label = '18-month publication')
    
    # Annotate each point with its frequency
    for i, freq in enumerate(year_counts.values):
        plt.text(year_counts.index[i], freq + 5, str(freq), ha='center')  # Adjusting y-offset for better visibility
    
    plt.show()
    
    
    #%% 1-2. 국가별 시계열 출원 동향 (USPTO)
    
    from copy import copy
    import seaborn as sns
    import matplotlib.pyplot as plt
    from collections import Counter 
    
    data_input = copy(data_preprocessed.loc[data_preprocessed['year_application'] <= 2023, :].reset_index(drop= 1))
    data_input = copy(data_input.loc[data_input['year_application'] >= 2001, :].reset_index(drop= 1)) 
    
    c = Counter(data_input['applicant_country_rep'])
    c = c.most_common(5)
    top5_country_list = [i[0] for i in c]
    data_input['applicant_country_rep'] = data_input['applicant_country_rep'].apply(lambda x : x if x in top5_country_list else np.nan)
    
    year_counts = data_input.groupby('applicant_country_rep')['year_application'].value_counts().sort_index().reset_index()
    year_counts['year_application'] = year_counts['year_application'].apply(lambda x : str(x))
    year_counts = year_counts.sort_values('year_application')
    
    
    total_measurements = year_counts.groupby('year_application')['count'].sum().reset_index()    
    total_measurements['applicant_country_rep'] = 'Total'
    
    
    plt.figure(figsize=(12, 6))
    sns.lineplot(data = year_counts,
                 x='year_application', 
                 y='count', 
                 hue = 'applicant_country_rep',
                 marker='o',)
    
    
    sns.barplot(data=total_measurements, x='year_application', 
                y='count', 
                hue='applicant_country_rep', palette='Greys')
    
    
    
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
    
    #%% 1-3. 수집 데이터 별 출원 동향
    from copy import copy
    import seaborn as sns
    import matplotlib.pyplot as plt
    import re
    
    data_input = copy(data_preprocessed.loc[data_preprocessed['year_application'] <= 2023, :].reset_index(drop= 1))
    data_input = copy(data_input.loc[data_input['year_application'] >= 2001, :].reset_index(drop= 1)) 
    
    data_input['file'] = data_input['file'].apply(lambda x : x.split('.')[0])
    
    year_counts = data_input.groupby('file')['year_application'].value_counts().sort_index().reset_index()
    
    year_counts['year_application'] = year_counts['year_application'].apply(lambda x : str(x))
    
    year_counts = year_counts.sort_values(by = ['year_application'])

    # year_counts = pd.concat([year_counts, total_measurements], ignore_index=True)
    
    plt.figure(figsize=(12, 6))
    sns.lineplot(data = year_counts,
                 x='year_application', 
                 y='count', 
                 hue = 'file',
                 marker='o',)
    
    plt.xlabel('Year',fontsize = 15)
    plt.ylabel('Annual Counts',fontsize = 15)
    plt.title('Patent Application Trend',fontsize = 15)
    
    plt.axvline(x = '2022', color = 'r', label = '18-month publication')
    
    # Annotate each point with its frequency
    for group in set(year_counts['file']) : 
        # if group not in ["HBM", "Neuromorphic", "NPU", "CXL"] : continue
        year_counts_group = year_counts.loc[year_counts['file'] == group, :]
        
        for i, freq in enumerate(year_counts_group['count']):
            plt.text(i, freq + 5, str(freq), ha='center')  # Adjusting y-offset for better visibility
        
    plt.show()
    
    
    #%% 2-1. CPC 비중 

    from collections import Counter
    
    import plotly.express as px
    import plotly.io as pio
    pio.renderers.default='browser'
    
    data_input = copy(data_preprocessed)
    
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
    fig.update_traces(textinfo="label+percent parent")
    fig.update_layout(margin=dict(t=0, b=0, l=0, r=0))                  
    fig.show()
    
    #%% 2-2. CPC-기업 비중

    pio.renderers.default='browser'    
    data_input = copy(data_preprocessed)
    data_input = data_input.loc[pd.isnull(data_input['applicant_rep_icon']) == False , :].reset_index(drop= 1)
    
    c = Counter(data_input['applicant_rep_icon'])
    
    temp = c.most_common(20)
    
    temp=[i[0] for i in temp]
    
    data_input = data_input.loc[data_input['applicant_rep_icon'].isin(temp), : ]
    data_input['CPC_sc'] = data_input['CPC_sc'].apply(lambda x : list(set(x)))
    data_input = data_input.explode("CPC_sc")
    
    df = data_input.groupby(['applicant_rep_icon','CPC_sc']).size()
    df = df.reset_index(drop = 0)
    
    
    fig = px.sunburst(df, path=['CPC_sc','applicant_rep_icon',], values= 0,
                      labels= 0,
                      maxdepth = 3,
                      color='CPC_sc', 
                      # color_discrete_map={'(?)':'black', 'HBM':'blue', '뉴로모픽':'red', 'NPU' : 'green', "PIM" : "purple"}
                      # hover_data=['iso_alpha'],
                      # color_continuous_scale='RdBu',
                      # color_continuous_midpoint=np.average(df['lifeExp'], weights=df['frequency']))
                      )
    
    fig.update_layout(font=dict(size=48))
    fig.update_traces(textinfo="label+percent parent")
    fig.update_layout(margin=dict(t=0, b=0, l=0, r=0))                  
    fig.show()
    
    #%% 2-3. 기업-CPC 비중

    pio.renderers.default='browser'    
    data_input = copy(data_preprocessed)
    data_input = data_input.loc[pd.isnull(data_input['applicant_rep_icon']) == False , :].reset_index(drop= 1)
    
    c = Counter(data_input['applicant_rep_icon'])
    
    temp = c.most_common(20)
    
    temp=[i[0] for i in temp]
    
    data_input = data_input.loc[data_input['applicant_rep_icon'].isin(temp), : ]
    data_input['CPC_sc'] = data_input['CPC_sc'].apply(lambda x : list(set(x)))
    data_input = data_input.explode("CPC_sc")
    
    df = data_input.groupby(['applicant_rep_icon','CPC_sc']).size()
    df = df.reset_index(drop = 0)
    
    
    fig = px.sunburst(df, path=['applicant_rep_icon','CPC_sc',], values= 0,
                      labels= 0,
                      maxdepth = 3,
                      color='applicant_rep_icon', 
                      # color_discrete_map={'(?)':'black', 'HBM':'blue', '뉴로모픽':'red', 'NPU' : 'green', "PIM" : "purple"}
                      # hover_data=['iso_alpha'],
                      # color_continuous_scale='RdBu',
                      # color_continuous_midpoint=np.average(df['lifeExp'], weights=df['frequency']))
                      )
    
    fig.update_layout(font=dict(size=48))
    fig.update_traces(textinfo="label+percent parent")
    fig.update_layout(margin=dict(t=0, b=0, l=0, r=0))                  
    fig.show()



    #%% 3. 출원인 TOP-N
    
    data_input = copy(data_preprocessed.loc[data_preprocessed['year_application']<= 2023, :].reset_index(drop= 1))   
    data_input = data_input.loc[pd.isnull(data_input['applicant_rep_icon']) == False , :].reset_index(drop= 1)

    c = Counter(data_input['applicant_rep_icon'])
    c = c.most_common(10)
    c = [i[0] for i in c]
    
    data_input['applicant_rep_icon'] = [i if i in c else 'ETC' for i  in data_input['applicant_rep_icon']  ] 
    data_input = data_input.loc[data_input['applicant_rep_icon'] != 'ETC', :].reset_index(drop = 1)
    
    sns.set_style('whitegrid')
    plt.figure(figsize=(6,12), dpi = 1000)
    
    # c.append("ETC")
    plot = sns.catplot(x="applicant_rep_icon", kind="count", palette="ch:1", 
                       data=data_input, 
                       order = c)
    
    for i, bar in enumerate(plot.ax.patches) : 
        
        h = bar.get_height()
        
        plot.ax.text(i, 
                     h+10, 
                     '{}'.format(int(h)),
                     ha= 'center',
                     va = 'center',
                     fontweight = 'bold',
                      # rotation = 90,
                     size = 12,)
        
        plot.ax.tick_params(axis='x', rotation=90)
    
    
    
    