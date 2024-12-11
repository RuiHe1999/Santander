# 1. package
import re
import numpy as np
import pandas as pd
from tqdm import tqdm 

import spacy
spacy.prefer_gpu()
snlp = spacy.load('es_core_news_lg')

import nltk
from nltk.corpus import stopwords
stpw = stopwords.words('spanish')

import stanza 
lang = 'es'
stanza_nlp = stanza.Pipeline(lang, 
                      processors="tokenize,pos,constituency", 
                      tokenize_no_ssplit=True,
                      download_method=None)

import warnings
warnings.filterwarnings("ignore")

import networkx as nx 
import antropy as ant
from scipy import stats

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
upos = ['ADJ', 'ADP', 'ADV', 'AUX', 'CCONJ', 'DET', 'INTJ', 'NOUN', 'NUM', 
        'PART', 'PRON', 'PROPN', 'PUNCT', 'SCONJ', 'SYM', 'VERB', 'X']
udep = [
    'acl', 'advcl', 'advmod', 'amod', 'appos', 'aux', 'case', 'cc', 'ccomp', 
    'clf', 'compound', 'conj', 'cop', 'csubj', 'dep', 'det', 'discourse', 
    'dislocated', 'expl', 'fixed', 'flat', 'goeswith', 'iobj', 'list', 
    'mark', 'nmod', 'nsubj', 'nummod', 'obj', 'obl', 'orphan', 'parataxis', 
    'punct', 'reparandum', 'root', 'vocative', 'xcomp'
]

utense = ['Fut', 'Imp', 'Past', 'Pqp', 'Pres']
uaspect = ['Hab', 'Imp', 'Iter', 'Perf', 'Prog', 'Prosp']
umood = ['Ind', 'Imp', 'Sub', 'Cnd']
ugender = ['Masc', 'Fem']

syn_feats = ['WordNum', 'Sent_Num', 'NP_Num']
syn_feats.append('WordTTR') 
syn_feats.append('stpw_ratio')
syn_feats += [x+'_PosR' for x in upos]
syn_feats += [x+'_DepR' for x in udep]
syn_feats.extend(['ADD', 'MDD'])
syn_feats += [x+'_TenseR' for x in utense]
syn_feats += ['Person1R', 'Person2R', 'Person3R',]
syn_feats += ['Negation']
syn_feats += [x+'_AspR' for x in uaspect]
syn_feats += [x+'_MoodR' for x in umood]
syn_feats += ['MascR', 'FemR',]
syn_feats += ['SingR', 'PlurR',]
syn_feats += ['DefiniteR', 'IndefiniteR',]
syn_feats.extend(['NodeNum', 'PhraseRatio', 'DepthMean', 'Depth', 'DepthNorm', 'DepthApEn', 'DepthSkew', 'Linearization'])
    
           
    
# 3. functions
def nltk_tree_to_stanza_tree(nltk_tree):

    if isinstance(nltk_tree, str):
        # Base case: a leaf node (i.e. a word)
        return stanza.models.constituency.parse_tree.Tree(nltk_tree, [])
    else:
        # Recursive case: a non-leaf node (i.e. a constituent)
        children = [nltk_tree_to_stanza_tree(child) for child in nltk_tree]
        return stanza.models.constituency.parse_tree.Tree(nltk_tree.label(), children)
    
def compute_syntax_tree(sent):

    # process the sentence 
    doc = stanza_nlp(sent)
    
    # get parsing string
    assert (len([sent for sent in doc.sentences]) == 1)
    input_str = str(doc.sentences[0].constituency)

    # remove punctionations
    input_str = re.sub(r' \(\s*PUNCT\s+[^)]*?\)', '', input_str)

    # removing nodes without children node
    input_str = re.sub(r' \([^() ]+\)', '', input_str)
    input_str = re.sub(r'\([^() ]+\)', '', input_str)
    
    # remove the ROOT node
    input_str = input_str[6:-1]
    
    # return empty string if nothing left 
    if input_str == '':
        return ''
    
    # Parse the input string and convert to an NLTK tree
    nltk_tree = nltk.Tree.fromstring(input_str)

    # binarize the nltk tree
    nltk_tree.chomsky_normal_form()
    
    # convert to a stanza tree
    # This is not necessary if you know how to parse the nltk tree into directed graph
    # unfortunately I only know how to convert stanza tree into directed graph
    # so I first convert the nltk tree to stanza tree 

    tree = nltk_tree_to_stanza_tree(nltk_tree)

    # convert the tree to a directed graph using NetworkX
    graph = nx.DiGraph()
    labels = {}

    def add_nodes_edges(tree, parent_node=None):
        # add node to graph
        node = len(graph)
        graph.add_node(node)

        # add label for node
        if isinstance(tree, stanza.models.constituency.parse_tree.Tree):
            label = tree.label
        else:
            label = tree
        labels[node] = label

        # add edge to parent node, if present
        if parent_node is not None:
            graph.add_edge(parent_node, node)

        # add child nodes and edges
        if isinstance(tree, stanza.models.constituency.parse_tree.Tree):
            for child in tree.children:
                # omit the last layer of words 
                if len(child.children) > 0:
                    add_nodes_edges(child, node)
                # add_nodes_edges(child, node)

    # recursively use the function to parse all nodes            
    add_nodes_edges(tree)

    return labels, graph


def compute_syntax_features(sent):
    
    # compute syntax tree
    labels, G = compute_syntax_tree(sent)
                                    
    # topological features            
    nodes = G.number_of_nodes()

    # leaves
    syn_leaves = [node for node in G.nodes if len(list(G.successors(node)))==0]
    leaf_count = len(syn_leaves)

    # phrasal nodes
    phrasal_count = nodes - leaf_count
    phrasal_leaf = phrasal_count / leaf_count
    
    
    # travel distance 
    td = [len(nx.shortest_path(G, source=0, target=node))-1 for node in syn_leaves]
    depth_apen = ant.app_entropy(td) if len(td) > 2 else 0
    depth_skew = stats.skew(td)
    
    # linearization
    line = stats.spearmanr(td, list(range(len(syn_leaves)))).statistic
    
    features = [nodes, phrasal_leaf, np.mean(td), np.max(td), np.max(td) / leaf_count, depth_apen, depth_skew, line]
    
    return features 

def extract_morphosyntax(text):
    global sub, i
    
    result = []
    
    # preprocess text 
    doc = snlp(text)
    sents = [sent.text for sent in doc.sents if sent.text.strip() != '']
    words = [token.text.lower() for token in doc if token.pos_ not in ['SPACE', 'PUNCT']]
    noun_chunks = [nph.text for nph in doc.noun_chunks]

    # quantity
    result.append(len(words))
    result.append(len(sents))
    result.append(len(noun_chunks))
    result.append(len(set(words))/len(words))

    # stopword propotion
    result.append(sum(1 for word in words if word in stpw) / len(words))

    # pos porpotion 
    pos_tags = [token.pos_ for token in doc]
    for pos in upos:
        result.append(pos_tags.count(pos) / len(pos_tags))

    # dependency as a list of triplets
    dependency = [(token.i, token.head.i, token.dep_) for token in doc]

    # dependency type porpotion 
    dep_types = [dep[2] for dep in dependency]
    for dep_type in udep:
        result.append(dep_types.count(dep_type) / len(dep_types)) 
       
    # dependency distance  
    dep_dist = [np.abs(dep[0] - dep[1]) for dep in dependency if dep[2]!='punct']   
    result.append(np.nanmean(dep_dist))
    result.append(np.nanmax(dep_dist))

    # tense
    tense = sum([token.morph.get('Tense') for token in doc], [])
    for ten in utense:
        result.append(tense.count(ten)/len(words)) 
       
    # person 
    person = sum([token.morph.get('Person') for token in doc], [])
    result.append(person.count('1') / len(person) if len(person) > 0 else 0) 
    result.append(person.count('2') / len(person) if len(person) > 0 else 0) 
    result.append(person.count('3') / len(person) if len(person) > 0 else 0) 

    # negation 
    negation = sum([token.morph.get('Polarity') for token in doc], [])
    result.append(negation.count('Neg')/len(words))

    # aspect
    aspect = sum([token.morph.get('Aspect') for token in doc], [])
    for asp in uaspect:
        result.append(aspect.count(asp)/len(words)) 
        
      
    # mood
    mood = sum([token.morph.get('Mood') for token in doc], [])
    for moo in umood:
        result.append(mood.count(moo)/len(words))

    # gender
    gender = sum([token.morph.get('Gender') for token in doc], [])
    result.append(gender.count('Masc') / len(words))
    result.append(gender.count('Fem') / len(words))

    # number
    number = sum([token.morph.get('Number') for token in doc], [])
    result.append(number.count('Sing') / len(words))
    result.append(number.count('Plur') / len(words))

    # definite
    definite = sum([token.morph.get('Definite') for token in doc], [])
    result.append(definite.count('Def') / len(words))
    result.append(definite.count('Ind') / len(words))

    # syntax tree
    cons_feats = []
    for sent in sents:
        try:
            cons_feats.append(compute_syntax_features(sent))
        except:
            print(f'Not parsable: {sub}, task {i}, {sent}')
    cons_feats = np.nanmean(cons_feats, axis=0)
    result.extend(cons_feats.tolist())
    
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
            result.extend(extract_morphosyntax(sub2text[sub]))
        else:
            pics = np.array([extract_morphosyntax(sub2text[sub][0]), 
                              extract_morphosyntax(sub2text[sub][1]), 
                              extract_morphosyntax(sub2text[sub][2])])
            result.extend(np.nansum(pics[:, :3], axis=0).tolist() + \
                          np.nanmean(pics[:, 3:], axis=0).tolist())
                
        result_.append(result)
                
        
    whole = [sub, 'whole'] + \
            np.nansum([r[2:5] for r in result_], axis=0).tolist() + \
            np.nanmean([r[5:] for r in result_], axis=0).tolist()
    result_.append(whole)
    
    results.extend(result_)
        
        
results_df = pd.DataFrame(results, columns=['PAR', 'Task'] + syn_feats)
results_df.to_csv('Features/morphosyntax.csv', index=False)  

















