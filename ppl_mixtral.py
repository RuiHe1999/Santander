# 1. package
import numpy as np
import pandas as pd
from tqdm import tqdm 

import spacy
spacy.prefer_gpu()
snlp = spacy.load('es_core_news_lg')

import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "ecastera/eva-mistral-dolphin-7b-spanish"
mistral_tokenizer = AutoTokenizer.from_pretrained(model_name)
mistral_model = AutoModelForCausalLM.from_pretrained(model_name)
mistral_model = mistral_model.to(device)
  
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


# 3. functions
def compute_perplexity(sentence):

    # Encode the sentence using the tokenizer
    input_ids = mistral_tokenizer.encode(sentence, return_tensors='pt').to(device)
    loss = mistral_model(input_ids, labels=input_ids).loss
    mistral_ppl = np.exp2(loss.item())
    
    return mistral_ppl

def extract_feature(text):
    text = text.strip()
    text = text.replace('...', '.')
    
    ppls = [compute_perplexity(text)]
                
    return ppls

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
            result.extend(extract_feature(sub2text[sub]))
        else:
            pics = np.array([extract_feature(sub2text[sub][0]), 
                              extract_feature(sub2text[sub][1]), 
                              extract_feature(sub2text[sub][2])])
            result.extend(np.nanmean(pics, axis=0).tolist())
                        
        result_.append(result)
        
    whole = [sub, 'whole'] + np.nanmean([r[2:] for r in result_], axis=0).tolist()
    result_.append(whole)
    
    results.extend(result_)
    
results_df = pd.DataFrame(results,  columns=['PAR', 'Task', 'ppl'])
results_df.to_csv('Features/ppl_mixtral.csv', index=False)   















