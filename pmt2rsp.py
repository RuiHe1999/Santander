# 1. packages 
import numpy as np
import pandas as pd
from tqdm import tqdm 

import torch
import torch.nn.functional as F

import clip
from PIL import Image
from transformers import AutoModel, AutoTokenizer
from multilingual_clip import pt_multilingual_clip

# 2. consonants
# CLIP model 
clip_model, clip_preprocess = clip.load("ViT-L/14")

tokenizer = AutoTokenizer.from_pretrained('hiiamsid/sentence_similarity_spanish_es')
model = AutoModel.from_pretrained('hiiamsid/sentence_similarity_spanish_es')

# Load Model & Tokenizer
pt_model = pt_multilingual_clip.MultilingualCLIP.from_pretrained('M-CLIP/XLM-Roberta-Large-Vit-L-14')
pt_tokenizer = AutoTokenizer.from_pretrained('M-CLIP/XLM-Roberta-Large-Vit-L-14')

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
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def clip_sim(text, picture):
    # get text features
    text_features = pt_model.forward([text], pt_tokenizer)

    # get image features
    image = [clip_preprocess(Image.open(picture).convert("RGB"))]
    image_input = torch.tensor(np.stack(image))
    image_feature = clip_model.encode_image(image_input).float()[0]

    return torch.nanmean(F.cosine_similarity(text_features, image_feature)).item()


def sent_sim(sentences):
    # Tokenize sentences 
    encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
    
    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)

    #   Perform pooling. In this case, max pooling.
    embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    
    #  normalize embeddings
    embeddings = F.normalize(embeddings, p=2, dim=1)
    scores = embeddings[:1] @ embeddings[1:].T
    
    return scores.item()

# 4. commands
# 4.0 preprocess data
# task 1: concatenate the response 
task1 = task1.sort_values(by=['ID', 'Question', 'Prompt', 'Interview'])
task1_intr = task1[task1['Interview'] == 'Interviewer']
task1_inte = task1[task1['Interview'] == 'Interviewee']

task1_text = {} 
for sub in subs:
    task1_text[sub] = []
    prompts = task1_intr[task1_intr['ID']==sub]['Prompt'].tolist()
    for prompt in prompts:
        pmt_text = task1_intr[(task1_intr['ID']==sub) & (task1_intr['Prompt']==prompt)]['Response']
        rsp_text = task1_inte[(task1_inte['ID']==sub) & (task1_inte['Prompt']==prompt)]['Response']
        assert(pmt_text.size==rsp_text.size==1)
        task1_text[sub].append([pmt_text.values[0], rsp_text.values[0]])

# sort data 
task2 = task2.sort_values(by=['ID', 'Question', 'Prompt', 'Interview'])
task2_intr = task2[task2['Interview'] == 'Interviewer']
task2_inte = task2[task2['Interview'] == 'Interviewee']

task2_text = {} 
for sub in subs:
    task2_text[sub] = []
    prompts = task2_intr[task2_intr['ID']==sub]['Prompt'].tolist()
    for prompt in prompts:
        pmt_text = task2_intr[(task2_intr['ID']==sub) & (task2_intr['Prompt']==prompt)]['Response']
        rsp_text = task2_inte[(task2_inte['ID']==sub) & (task2_inte['Prompt']==prompt)]['Response']
        assert(pmt_text.size==rsp_text.size==1)
        task2_text[sub].append([pmt_text.values[0], rsp_text.values[0]])

# task 4: concatenate the response by picture
task4 = task4[task4['Interview']=='Interviewee']
task4 = task4.sort_values(by=['ID', 'Question', 'Prompt']) 

task4_text = {} 
for sub in subs:
    response1 = ' '.join(task4[(task4['ID']==sub) & (task4['Prompt'] == 1)]['Response'].values)
    response2 = ' '.join(task4[(task4['ID']==sub) & (task4['Prompt'] == 2)]['Response'].values)
    response3 = ' '.join(task4[(task4['ID']==sub) & (task4['Prompt'] == 3)]['Response'].values)
    task4_text[sub] = [[response1, 'pic_1.jpg'], 
                        [response2, 'pic_2.jpg'], 
                        [response3, 'pic_3.jpg'],]

# task 7: only analyze retell
task7_rt = task7[task7['Prompt']=='retell']
task7_rd = task7[task7['Prompt']=='read']
task7_text = {} 
for sub in subs:
    task7_text[sub] = [[task7_rd[task7_rd['ID']==sub]['Response'].values.item(),
                        task7_rt[task7_rt['ID']==sub]['Response'].values.item()]]

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
            result.append(np.mean([sent_sim(t) for t in sub2text[sub]]))
        else:
            pics = np.array([clip_sim(sub2text[sub][0][0], sub2text[sub][0][1]),
                              clip_sim(sub2text[sub][1][0], sub2text[sub][1][1]), 
                              clip_sim(sub2text[sub][2][0], sub2text[sub][2][1])])
        
            result.append(np.nanmean(pics, axis=0).tolist())
        
        result_.append(result)
        
    whole = [sub, 'whole'] + \
            np.nanmean([r[2:] for r in result_], axis=0).tolist()
    result_.append(whole)
    
    results.extend(result_)

results_df = pd.DataFrame(results,  columns=['PAR', 'Task', 'pmt2rsp'])
results_df.to_csv('Features/pmt2rsp.csv', index=False)  

