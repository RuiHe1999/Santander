# 1. package
import re
import numpy as np
import pandas as pd
from tqdm import tqdm 

import spacy
spacy.prefer_gpu()
snlp = spacy.load('es_core_news_lg')

from pysentimiento import create_analyzer
sentiment_analyzer = create_analyzer(task="sentiment", lang="es")
hate_speech_analyzer = create_analyzer(task="hate_speech", lang="es")

# 2. consonants
# read data
task1 = pd.read_excel('Texts/Task_1.xlsx')
task2 = pd.read_excel('Texts/Task_2.xlsx')
task4 = pd.read_excel('Texts/Task_4.xlsx')
task7 = pd.read_excel('Texts/Task_7.xlsx')
assert(sorted(task1['ID'].unique()) == sorted(task2['ID'].unique()) == sorted(task4['ID'].unique()) == sorted(task7['ID'].unique()))

# read demograph files
demo = pd.read_spss('Data_Santander_Discouse.sav')
demo = demo[demo['MF_Group'].isin(['Paciente', 'Control'])]
demo['MF_Group'] = demo['MF_Group'].cat.remove_unused_categories()
demo['codigoFamiliar'] = demo['codigoFamiliar'].apply(lambda x: f'DS_{x}')
demo = demo[demo['codigoFamiliar'].isin(task1['ID'].unique())]

# subjects
subs = sorted(demo['codigoFamiliar'].values)
NumSubs = len(subs)

# features
senti_feats = ['NeutralR', 'NegativeR', 'PostiveR', 
               'HatefulR', 'AggressiveR', 'TargetedR']

# 3. functions

def extract_sentiment(text):
    global sub, i
    
    result = []

    # preprocess text 
    doc = snlp(text)
    sents = [sent.text for sent in doc.sents if sent.text.strip() != '']

    # sentiment analysis 
    sentiment_results = [sentiment_analyzer.predict(sent).output for sent in sents]
    for sentiment_label in ['NEU', 'NEG', 'POS']:
        result.append(sentiment_results.count(sentiment_label) / len(sents))

    # hate speech analysis 
    hate_speech_results = sum([hate_speech_analyzer.predict(sent).output for sent in sents], [])
    for hate_speech_label in ['hateful', 'aggressive', 'targeted']:
        result.append(hate_speech_results.count(hate_speech_label) / len(sents))

    return result

# 4. commands
# 4.0 preprocess data
# task 1: concatenate the response 
task1 = task1[task1['Interview']=='Interviewee']
task1 = task1.sort_values(by=['ID', 'Question', 'Prompt']) 

task1_text = {} 
for sub in subs:
    response = ' '.join(task1[task1['ID']==sub]['Response'].values)
    task1_text[sub] = response

# task 2: concatenate the response 
task2 = task2[task2['Interview']=='Interviewee']
task2 = task2.sort_values(by=['ID', 'Question', 'Prompt']) 

task2_text = {} 
for sub in subs:
    response = ' '.join(task2[task2['ID']==sub]['Response'].values)
    task2_text[sub] = response

# task 4: concatenate the response by picture
task4 = task4[task4['Interview']=='Interviewee']
task4 = task4.sort_values(by=['ID', 'Question', 'Prompt']) 

task4_text = {} 
for sub in subs:
    response1 = ' '.join(task4[(task4['ID']==sub) & (task4['Prompt'] == 1)]['Response'].values)
    response2 = ' '.join(task4[(task4['ID']==sub) & (task4['Prompt'] == 2)]['Response'].values)
    response3 = ' '.join(task4[(task4['ID']==sub) & (task4['Prompt'] == 3)]['Response'].values)
    task4_text[sub] = [response1, response2, response3]

# task 7: only analyze retell
task7 = task7[task7['Prompt']=='retell']
task7_text = {} 
for sub in subs:
    task7_text[sub] = task7[task7['ID']==sub]['Response'].values.item()

# 4.1 extract features for text
results = []
for sub in tqdm(subs):
    result_ = []
    for i, sub2text in enumerate([task1_text, task2_text, task4_text, task7_text]):
        i = {0: 1, 1: 2, 2: 4, 3: 7}[i]
        
        result = []
        result.append(sub)
        if i != 7:
            result.append(f'Q{i}')
        else:
            result.append(f'Q{i}rt')
              
        if i != 4:
            result.extend(extract_sentiment(sub2text[sub]))
        else:
            pics = np.array([extract_sentiment(sub2text[sub][0]), 
                              extract_sentiment(sub2text[sub][1]), 
                              extract_sentiment(sub2text[sub][2])])
            result.extend(np.nanmean(pics, axis=0).tolist())
                
        result_.append(result)
    
    results.extend(result_)
        
        
results_df = pd.DataFrame(results, columns=['PAR', 'Task'] + senti_feats)
results_df.to_csv('Features/sentiment.csv', index=False)  




    
    





