# 1. package
import numpy as np
import pandas as pd
from tqdm import tqdm 

import torch
import torch.nn.functional as F

from util import (get_sbert_embedding, get_bert_embedding, consecK, 
                  global_similarity, compute_wave_features, 
                  centroid_stat, centroid_cuml)

import spacy
snlp = spacy.load('es_core_news_lg')

from gensim.models import fasttext
ft = fasttext.load_facebook_vectors('cc.es.300.bin')

from nltk.corpus import stopwords
stpw = stopwords.words('spanish')

from transformers import RobertaTokenizer, RobertaModel
bert_name = 'PlanTL-GOB-ES/roberta-large-bne'
bert_tokenizer = RobertaTokenizer.from_pretrained(bert_name)
bert_model = RobertaModel.from_pretrained(bert_name)

from transformers import AutoTokenizer, AutoModel
sbert_name = 'hiiamsid/sentence_similarity_spanish_es'
sbert_tokenizer = AutoTokenizer.from_pretrained(sbert_name)
sbert_model = AutoModel.from_pretrained(sbert_name)

import warnings
warnings.filterwarnings("ignore")
    
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

# feature name
wav_feats = ['MeanK1', 'MeanK2', 'MeanG', 'MCR', 'SSC', 'WL', 'Var', 'Peak', 'Valley', 
             'Amp', 'Skew', 'Kurt', 'ApEn', 'Acf1', 'AcfZcr']

# 3. functions
def compute_feats(embeds): 
    
    if len(embeds) <= 3:
        return [np.nan] * 3 * len(wav_feats)

    else:
        # consecutive and global semantic similarity 
        consec_feats = compute_wave_features(consecK(embeds, 1))
        consec_feats.insert(1, np.mean(consecK(embeds, 2)))
        consec_feats.insert(2, global_similarity(embeds))

        # similarity between each unit and the static centroid
        cent_stat = compute_wave_features(centroid_stat(embeds))
        cent_stat.insert(1, np.nan)
        cent_stat.insert(2, np.nan)

        # similarity between each unit and the cumulative centroids
        cent_cuml = compute_wave_features(centroid_cuml(embeds))
        cent_cuml.insert(1, np.nan)
        cent_cuml.insert(2, np.nan)
        
        data = consec_feats + cent_stat + cent_cuml 
    
    
    return data

def extract_feature(text):
    text = text.strip()
    text = text.replace('...', '.')
    
    # preprocess text 
    doc = snlp(text)
    sents = [sent.text for sent in doc.sents if sent.text.strip() != '']
    tokens = [token.text.lower() for token in doc
              if ((token.pos_ not in ['SPACE', 'PUNCT']) and (token.text.lower() not in stpw))]

    # vectorize tokens and sentences
    # ft_embeds = np.array([ft.get_word_vector(token) for token in tokens])
    ft_embeds =  F.normalize(torch.Tensor([ft[token] for token in tokens]), p=2, dim=1).numpy()
    bert_embeds = get_bert_embedding(sents, bert_tokenizer, bert_model)
    sent_embeds = get_sbert_embedding(sents, sbert_tokenizer, sbert_model)

    # extract features
    features = [len(tokens), len(sents)] + compute_feats(ft_embeds) + compute_feats(bert_embeds) + compute_feats(sent_embeds)
                
    return features

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
        print(i)
        
        result = []
        result.append(sub)
        if i != 7:
            result.append(f'Q{i}')
        else:
            result.append(f'Q{i}rt')
            
        if i != 4:
            result.extend(extract_feature(sub2text[sub]))
        else:
            pics = np.array([extract_feature(sub2text[sub][0]), 
                             extract_feature(sub2text[sub][1]), 
                             extract_feature(sub2text[sub][2])])
            result.extend(np.nansum(pics[:, :2], axis=0).tolist() + \
                          np.nanmean(pics[:, 2:], axis=0).tolist())
                        
        result_.append(result)
        
    whole = [sub, 'whole'] + \
            np.nansum([r[2:4] for r in result_], axis=0).tolist() + \
            np.nanmean([r[4:] for r in result_], axis=0).tolist()
    result_.append(whole)
    
    results.extend(result_)
    
results_df = pd.DataFrame(results, 
                          columns=['PAR', 'Task', 'TokenNum', 'SentNum'] + \
            [f'FT{tag}_{feat}' for tag in ['', '_stat', '_cuml'] for feat in wav_feats] + \
            [f'BERT{tag}_{feat}' for tag in ['', '_stat', '_cuml'] for feat in wav_feats] + \
            [f'Sent{tag}_{feat}' for tag in ['', '_stat', '_cuml'] for feat in wav_feats])

results_df = results_df.drop(columns=['FT_stat_MeanK2', 'FT_stat_MeanG', 
                                      'FT_cuml_MeanK2', 'FT_cuml_MeanG', 
                                      'BERT_stat_MeanK2', 'BERT_stat_MeanG', 
                                      'BERT_cuml_MeanK2', 'BERT_cuml_MeanG', 
                                      'Sent_stat_MeanK2', 'Sent_stat_MeanG', 
                                      'Sent_cuml_MeanK2', 'Sent_cuml_MeanG',])

results_df.to_csv('Features/wav_sim.csv', index=False)   










