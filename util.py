# import re
# import networkx as nx
import torch
import torch.nn.functional as F
from torch import Tensor

import numpy as np
import pandas as pd
import antropy as ant
from scipy import stats
from scipy.spatial import distance
from statsmodels.tsa.stattools import acf

def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

def get_sbert_embedding(sents, tokenizer, model):
    
    # Tokenize sentences
    encoded_input = tokenizer(sents, padding=True, truncation=True, return_tensors='pt')
    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)
    # Perform pooling
    embeddings = average_pool(model_output.last_hidden_state, encoded_input['attention_mask'])
    embeddings = F.normalize(embeddings, p=2, dim=1)
    
    return embeddings.numpy()
    
def get_bert_embedding(sents, tokenizer, model):
    
    embeddings = []
    for sent in sents:
        
        encoded_input = tokenizer(sent, return_tensors='pt', truncation=True)
        output, pooler_output = model(input_ids=encoded_input['input_ids'],
                                      attention_mask=encoded_input['attention_mask'],
                                      return_dict=False)
        embeddings.extend(output[0].tolist()) 
    
    embeddings = torch.Tensor(embeddings)
    embeddings = F.normalize(embeddings, p=2, dim=1)
        
    return embeddings.numpy()

def slope_sign_changes(data):
    
    ssc = 0
    for i in list(range(len(data)-1))[1:]:
        if (data[i] < data[i+1]) and (data[i] < data[i-1]):
            ssc += 1
        elif (data[i] > data[i+1]) and (data[i] > data[i-1]):
            ssc += 1
        else:
            ssc += 0
    return ssc

def consecK(embeddings, k):
    
    assert (k >= 1)
    
    # k order sem sim
    consecs_k = []
    for i in range(len(embeddings)-k):
        embed1 = embeddings[i]
        embed2 = embeddings[i+k]
        consec_k = 1 - distance.cosine(embed1, embed2)
        consecs_k.append(consec_k)
    
    return np.array(consecs_k)

def global_similarity(embeddings):

    mean_g = []
    for i in range(len(embeddings)):
        for j in range(len(embeddings)):
            if i< j:
                embed1 = embeddings[i]
                embed2 = embeddings[j]
                mean_g.append(1 - distance.cosine(embed1, embed2))
        
    return np.mean(mean_g)
       
def centroid_stat(embeddings):
    
    centroid = np.mean(embeddings, axis=0)
    
    stat_cent_sims = []
    for i in range(len(embeddings)):
        embed = embeddings[i]
        stat_cent_sim = 1 - distance.cosine(embed, centroid)
        stat_cent_sims.append(stat_cent_sim)
    
    return np.array(stat_cent_sims)
    
def centroid_cuml(embeddings):

    cuml_cent_sims = []
    for i in range(1, len(embeddings)):
        embed = embeddings[i]
        centroid = np.mean(embeddings[:i, :], axis=0)
        cuml_cent_sim = 1 - distance.cosine(embed, centroid)
        cuml_cent_sims.append(cuml_cent_sim)
    
    return np.array(cuml_cent_sims)

def subject_sim(embeddings):
       
    # k order sem sim
    sims = []
    for i in range(1, len(embeddings)):
        embed1 = embeddings[0]
        embed2 = embeddings[i]
        sim = 1 - distance.cosine(embed1, embed2)
        sims.append(sim)
    
    return np.array(sims)

def compute_wave_features(sim_wav):
        
    assert(np.all(sim_wav == sim_wav))
       
    # mean
    mean = np.mean(sim_wav)
    
    # variance
    var = np.var(sim_wav)
    # peak
    peak = np.max(sim_wav)
    # valley
    valley = np.min(sim_wav)
    # amplitude
    amplitude = peak - valley
    # skewness
    skew = stats.skew(sim_wav)
    # excess kurtosis
    kurt = stats.kurtosis(sim_wav) - 3
    # mean crossing rate
    mcr = ((np.diff(np.sign(sim_wav-np.mean(sim_wav))) != 0).sum() - ((sim_wav-np.mean(sim_wav)) == 0).sum()) / (len(sim_wav) - 1)
    # normalized slope sign changes
    ssc = slope_sign_changes(sim_wav) / (len(sim_wav) - 2)
    # waveform length
    wl = np.mean([np.abs(sim_wav[i+1] - sim_wav[i]) for i in range(len(sim_wav)-1)])
    # entropy
    apen = ant.app_entropy(sim_wav)
    # ACW
    acw = acf(sim_wav, nlags=len(sim_wav)-1, qstat=False, alpha=None, fft=False)
    acw_zcr = ((np.diff(np.sign(acw)) != 0).sum() - (acw == 0).sum()) / (len(acw) - 1)
    
    feature = [mean, mcr, ssc, wl, var, peak, valley, amplitude, skew, kurt, apen, acw[1], acw_zcr]
     
    return feature




    

    
    








