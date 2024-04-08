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
    directory += '우주/'
    
    file_list = os.listdir(directory)
    
    # 데이터 로드 
    with open(directory + 'data_preprocessed.pkl', 'rb') as file:  
        data_preprocessed = pickle.load(file)  
        
    
    #%% 1. 시계열 출원 동향 
    
    from copy import copy
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    data_input = copy(data_preprocessed.loc[data_preprocessed['year_application'] <= 2023, :].reset_index(drop= 1))
    
    year_counts = data_input['year_application'].value_counts().sort_index()
    
    plt.figure(figsize=(12, 6))
    sns.lineplot(x=year_counts.index, y=year_counts.values, marker='o', color='green', label = 'count')
    
    plt.xlabel('Year',fontsize = 15)
    plt.ylabel('Annual Frequency',fontsize = 15)
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
    
    #%% 2. CPC 비중 

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
    
    
    
    