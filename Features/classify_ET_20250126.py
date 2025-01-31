###############################################################################
# 1. packages# ################################################################
###############################################################################
import numpy as np
random_state = 42
np.random.seed(random_state)

import pandas as pd
from tqdm import tqdm, trange

from sklearn import ensemble
from sklearn.decomposition import PCA
from sklearn.preprocessing  import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from feature_engine.selection import DropCorrelatedFeatures

import factor_analyzer
import pingouin as pg
import scikit_posthocs as sp
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats import multitest

import seaborn as sns
from matplotlib import pyplot as plt
plt.rcParams['font.family'] = 'Arial'

import warnings
warnings.filterwarnings('ignore')

###############################################################################
# 2. consonants (with data preparation) #######################################
###############################################################################
out_dir = 'results/'

# read demograph files
demo = pd.read_spss('Data_Santander_Discouse.sav')
demo = demo[demo['MF_Group'].isin(['Paciente', 'Control'])]
demo['MF_Group'] = demo['MF_Group'].cat.remove_unused_categories().map({'Control':0, 'Paciente':1})
demo['codigoFamiliar'] = demo['codigoFamiliar'].apply(lambda x: f'DS_{x}')

# read acoustic data
ege = pd.read_csv('eGeMAPs.csv')
prosogram = pd.read_csv('Prosogram_results.csv')
acoustic = pd.merge(prosogram, ege,  on=['PAR', 'Task'], how='inner')
acoustic = acoustic.rename(columns=lambda x: x.replace('-', '_').replace('.', '_'))
acoustic_feats = acoustic.columns[2:].tolist()

# read semantic data
semantic = pd.merge(pd.read_csv('wav_sim.csv'),
                    pd.read_csv('pmt2rsp.csv'),  
                    on=['PAR', 'Task'], how='inner')
semantic = pd.merge(semantic, pd.read_csv('ppl_mixtral.csv'), on=['PAR', 'Task'], how='inner')
semantic = semantic.drop(columns=['TokenNum', 'SentNum'])
semantic_feats = semantic.columns[2:].tolist()

# read morphosyntax data
morphosyntax = pd.merge(pd.read_csv('wav_sim.csv')[['PAR', 'Task', 'TokenNum']],
                              pd.read_csv('morphosyntax.csv'),  
                              on=['PAR', 'Task'], how='inner')
morphosyntax['SentNum'] = morphosyntax['Sent_Num'].copy()
morphosyntax = morphosyntax.drop(columns=['punct_DepR', 'Sent_Num'])
morphosyntax_feats = morphosyntax.columns[2:]
morphosyntax_feats = [feat for feat in morphosyntax_feats 
                      if len(morphosyntax[feat].unique()) != 1]

morphosyntax = morphosyntax[['PAR', 'Task'] + morphosyntax_feats]

# Concatenate the dataframes based on ID and task number 
feature_merge = pd.merge(acoustic, semantic, on=['PAR', 'Task'], 
                          suffixes=('_acou', '_semt'), 
                          how='left')
feature_merge = pd.merge(feature_merge, morphosyntax, on=['PAR', 'Task'], 
                          suffixes=('', '_msyn'), 
                          how='left')

# into a dictionary
# shuffle the order of features 
acoustic_feats = np.random.default_rng(seed=random_state).permutation(acoustic_feats).tolist()
semantic_feats = np.random.default_rng(seed=random_state).permutation(semantic_feats).tolist()
morphosyntax_feats = np.random.default_rng(seed=random_state).permutation(morphosyntax_feats).tolist()

all_features = semantic_feats + morphosyntax_feats + acoustic_feats
all_features = np.random.default_rng(seed=random_state).permutation(all_features).tolist()

dict_feat = {
    'acoustic': acoustic_feats,
    'semantic': semantic_feats,
    'morphsyn': morphosyntax_feats,
    'all': all_features
    }

# tasks
tasks = ['1', '2', '4', '7rd', '7rt']

###############################################################################
# 3. functions ################################################################
###############################################################################
find_k = lambda dic: min([key for key, value in dic.items() 
                          if value == max(dic.values())])

def split_task_data(df):
    """
    Splits the input dataframe into training, testing, and validation subsets for each fold
    based on the global cross-validation split configuration.

    Parameters
    ----------
    df : DataFrame
        The input dataframe containing all relevant data. 
        Must include a 'PAR' column with the IDs.

    Returns
    -------
    splits : list of DataFrame
        A list of DataFrames, where each DataFrame corresponds to a fold.
        Each DataFrame has an additional 'subset' column with values:
            - 'train': Subjects belonging to the training subset for this fold.
            - 'test': Subjects belonging to the testing subset for this fold.
            - 'val': Subjects belonging to the validation subset for this fold.
    """
    
    global kfold, dict_folds

    splits = []
    for fold in range(kfold):
        data = df.copy()
        data.insert(1, 'subset', 'None')
        data.loc[data.PAR.isin(dict_folds['train'][fold]), 'subset'] = 'train'
        data.loc[data.PAR.isin(dict_folds['test'][fold]), 'subset'] = 'test'
        data.loc[data.PAR.isin(dict_folds['val'][fold]), 'subset'] = 'val'
        splits.append(data)
    
    return splits

def vote_result(predict_probas):
    """
    Perform majority voting based on predicted probabilities from multiple tasks.

    Parameters
    ----------
    predict_probas : np.array
        A NumPy array containing the predicted probabilities for each class.
        The array has shape (n_samples, n_classes), where:
        - n_samples is the number of data points.
        - n_classes is the number of target classes.
        Each row in the array represents the probability distribution 
        over all classes for a single sample.

    Returns
    -------
    vote_predict : np.array
        A 1D NumPy array of shape (n_samples,) containing the final predicted class 
        labels for each sample after majority voting. 
        - If there is a tie in votes and the number of tasks is even, the sum of predicted 
          probabilities across all tasks is used to break the tie.

    """
    
    
    predict_probas = np.array(predict_probas) 
    
    # count votes
    predicted_labels = np.array([np.argmax(proba, axis=1) for proba in predict_probas])
    vote_count = np.vstack([np.sum(predicted_labels==0, axis=0), 
                            np.sum(predicted_labels==1, axis=0)])

    
    if len(tasks) % 2 != 0: 
        # uses predicted class labels for majority rule voting
        vote_predict = np.argmax(vote_count, axis=0)
        
    else:
        # uses predicted class labels for majority rule voting
        vote_predict = np.argmax(vote_count, axis=0)
        # Handle ties: the argmax of the sums of the predicted probabilities
        tie_predict = np.where(vote_count[0] == vote_count[1])[0]
        vote_predict[tie_predict] = np.argmax(np.sum(predict_probas, axis=0), axis=1)[tie_predict]
   
    return vote_predict    
    
def find_best_k(data, feats, clf, fold_i, tasks=tasks):
    """
    

    Parameters
    ----------
    data : DataFrame
        The comprehensive dataframe with all linguistic features from every task
    feats : list
        The list of feature names.
    clf : sklearn.base.BaseEstimator
        An instance of a scikit-learn classifier that implements the `fit` and `predict` methods.
    fold_i : int
        Fold index for cross-validation.
    tasks : list, optional
        The list of tasks to evaluate. Defaults to `tasks`.

    Returns
    -------
    k2acc : dict
        keys: number of selected features; 
        values: a list of scores, PRFA of validation, PRFA of test

    """
    
    
    k2acc = {}
    for k in trange(1, len(feats)+1, desc=f'Fold {fold_i}: optimal k search'):
        
        # train test val split 
        test_data = data[data['subset']=='test']
        val_data = data[data['subset']=='val']
        train_data = data[data['subset']=='train']
           
        test_predict_probas = []
        val_predict_probas = []

        for task in tasks:
            X_train = train_data[train_data['Task']==f'Q{task}'][feats]
            y_train = train_data[train_data['Task']==f'Q{task}']['MF_Group']
            X_val = val_data[val_data['Task']==f'Q{task}'][feats]
            y_val = val_data[val_data['Task']==f'Q{task}']['MF_Group']
            X_test = test_data[test_data['Task']==f'Q{task}'][feats]
            y_test = test_data[test_data['Task']==f'Q{task}']['MF_Group']

            # set up a pipeline 
            model = Pipeline([
                # standard scaling
                ('scaler', StandardScaler()),       
                # impute nan with zero
                ('imputer', SimpleImputer(strategy='constant', fill_value=0)),     
                # remove correlated features 
                ('feature_removal_correlation', 
                  DropCorrelatedFeatures(variables=None, method='pearson', threshold=0.8)),  
                # selection based on F value                     
                ('feature_selection', SelectKBest(f_classif, k=k)),        
                # classifier
                ('classification', clf)                                      
            ])

            # train model to predict 
            model.fit(X_train, y_train)

            # test the model
            val_predict_probas.append(model.predict_proba(X_val))
            test_predict_probas.append(model.predict_proba(X_test))
            
        # majority voting
        val_pred = vote_result(val_predict_probas)
        test_pred = vote_result(test_predict_probas)
        
        # scores
        precision_val = precision_score(y_val, val_pred)
        recall_val = recall_score(y_val, val_pred)
        f1_val = f1_score(y_val, val_pred)
        acc_val = accuracy_score(y_val, val_pred)

        precision_test = precision_score(y_test, test_pred)
        recall_test = recall_score(y_test, test_pred)
        f1_test = f1_score(y_test, test_pred)
        acc_test = accuracy_score(y_test, test_pred)

        k2acc[k] = [precision_val, recall_val, f1_val, acc_val,
                    precision_test, recall_test, f1_test, acc_test]
        
    return k2acc
    
def map_feat_to_group(feat):
    """
    Maps a given feature name to its corresponding category.

    Parameters
    ----------
    feat : str
        The name of the feature to be mapped.

    Returns
    -------
    group : str
        The group category to which the feature belongs. Possible values are:
        - 'acoustics' for acoustic features.
        - 'semantics' for semantic features.
        - 'morphsyn' for morphosyntactic features.

    Raises
    ------
    ValueError
        If the feature does not belong to any of the predefined categories.
    """
    
    if feat in acoustic_feats:
        return 'acoustics'
    
    if feat in semantic_feats:
        return 'semantics'
    
    if feat in morphosyntax_feats:
        return 'morphsyn'
    
    # If the feature does not match any group, raise an error
    raise ValueError(f"The feature '{feat}' does not belong to any predefined group.")


###############################################################################
# 4. Data preparation #########################################################
###############################################################################
# 10 fold startified split 
kfold = 9

stratified_kfold = StratifiedKFold(n_splits=kfold, shuffle=True, random_state=random_state)

# test set, 4 from HC, 5 fron SZ
grouped_demo = demo.groupby('MF_Group')
sample_0 = grouped_demo.get_group(0).sample(n=4, random_state=random_state)
sample_1 = grouped_demo.get_group(1).sample(n=5, random_state=random_state)
test_demo = pd.concat([sample_0, sample_1], axis=0).codigoFamiliar

# shuffle the demographic data
shuffled_demo = demo[~demo['codigoFamiliar'].isin(test_demo)].sample(frac=1, random_state=random_state).reset_index(drop=True)

# stratified k fold
stratified_kfold_split = stratified_kfold.split(shuffled_demo.codigoFamiliar, shuffled_demo.MF_Group)

# save in a dictionary 
dict_folds = {'train': {}, 'test': {}, 'val': {}}
for fold, (train_index, test_index) in enumerate(stratified_kfold_split):
    dict_folds['train'][fold] = shuffled_demo.codigoFamiliar.iloc[train_index].tolist()
    dict_folds['val'][fold] = shuffled_demo.codigoFamiliar.iloc[test_index].tolist()
    dict_folds['test'][fold] = test_demo

# task data
task1 = feature_merge[feature_merge['Task']=='Q1']
task2 = feature_merge[feature_merge['Task']=='Q2']
task4 = feature_merge[feature_merge['Task']=='Q4']
task7rd = feature_merge[feature_merge['Task']=='Q7rd']
task7rt = feature_merge[feature_merge['Task']=='Q7rt']

# split task data 
task_merge_df  = pd.concat([task1, task2, task4, task7rd, task7rt])
task_merge_df = task_merge_df.merge(demo, left_on='PAR', right_on='codigoFamiliar', how='left')
task_merges = split_task_data(task_merge_df)


###############################################################################
# 5. classifier ###############################################################
###############################################################################
param = {'n_estimators': 5,           
          'max_depth': 5,
          'min_samples_split': 5,
          'min_samples_leaf': 2,
          'criterion': 'entropy',
          'random_state': random_state}
clf = ensemble.ExtraTreesClassifier(**param)

###############################################################################
# 6. machine learing within each fold #########################################
###############################################################################
k2acc_dfs = pd.DataFrame()
for i, task_merge in enumerate(task_merges):   
    k2acc = find_best_k(task_merge, all_features, clf, i+1)
    k2acc_df = pd.DataFrame.from_dict(k2acc, orient='index', 
                                      columns=['val_precision', 'val_recall', 'val_f1', 'val_acc',
                                               'test_precision', 'test_recall', 'test_f1', 'test_acc'])
    k2acc_df.reset_index(inplace=True)
    k2acc_df.rename(columns={'index': 'K'}, inplace=True)
    k2acc_df.insert(0, 'Fold', i+1)    
    k2acc_dfs = pd.concat([k2acc_dfs, k2acc_df])

k2acc_dfs['ag_acc'] = k2acc_dfs.apply( 
    lambda row: (row['test_acc'] + row['val_acc']) / (np.abs(row['test_acc'] - row['val_acc']) * 100) 
                if np.abs(row['test_acc'] - row['val_acc']) >= 0.05
                else (row['test_acc'] + row['val_acc']), 
    axis=1)

           
# find best k per fold and the averaged PRFA across 10 folds
fold_acc = []
for fold_i, fold_df in k2acc_dfs.groupby(['Fold']):
    max_k = find_k(dict(zip(fold_df.K, fold_df.ag_acc)))
    max_acc = fold_df[fold_df['K']==max_k]
    fold_acc.append(max_acc)

fold_acc = pd.concat(fold_acc)
fold_acc_mean = fold_acc.mean(axis=0)
fold_acc_mean['Fold'] = 'Avg'
fold_acc_mean['K'] = '/'
fold_acc = pd.concat([fold_acc, pd.DataFrame([fold_acc_mean])], axis=0)
fold_acc.to_csv(out_dir + 'results.csv', index=False)

###############################################################################
# 7. feature importance #######################################################
###############################################################################
# 7.1 compute feature importance score from the tree model
feature_importance = []
selected_features = []
for fold_i, task_merge in tqdm(enumerate(task_merges), total=kfold, desc='feat importance'): 

    # train test val split 
    test_data = task_merge[task_merge['subset']=='test']
    val_data = task_merge[task_merge['subset']=='val']
    train_data = task_merge[task_merge['subset']=='train']

    selected_features_ = []
    for task in tasks:
        X_train = train_data[train_data['Task']==f'Q{task}'][all_features]
        y_train = train_data[train_data['Task']==f'Q{task}']['MF_Group']
        X_val = val_data[val_data['Task']==f'Q{task}'][all_features]
        y_val = val_data[val_data['Task']==f'Q{task}']['MF_Group']
        X_test = test_data[test_data['Task']==f'Q{task}'][all_features]
        y_test = test_data[test_data['Task']==f'Q{task}']['MF_Group']

        # set up a pipeline
        fold_k = fold_acc[fold_acc['Fold'] == fold_i+1]['K'].item()
        
        # set up a pipeline 
        model = Pipeline([
            # standard scaling
            ('scaler', StandardScaler()),       
            # impute nan with zero
            ('imputer', SimpleImputer(strategy='constant', fill_value=0)),     
            # remove correlated features 
            ('feature_removal_correlation', 
              DropCorrelatedFeatures(variables=None, method='pearson', threshold=0.8)),  
            # selection based on F value                     
            ('feature_selection', SelectKBest(f_classif, k=fold_k)),        
            # classifier
            ('classification', clf)                                      
        ])
        
        # train model to predict 
        model.fit(X_train, y_train)
        
        # selected feature 
        feature_selector = model.named_steps['feature_selection']
        selected_feature_idx = feature_selector.get_support(indices=True)
        selected_feature = X_train.columns[selected_feature_idx] if hasattr(X_train, 'columns') else selected_feature_idx
        selected_features_.append(selected_feature)
        
        # feature importance
        selected_feature_importance = model.named_steps['classification'].feature_importances_
                
        # into a dataframe
        feature_summary = pd.DataFrame({'Feature': all_features, 'Importance': 0})
        importance_dict = dict(zip(selected_feature, selected_feature_importance))
        feature_summary['Importance'] = feature_summary['Feature'].map(importance_dict).fillna(0)
        
        feature_summary.insert(0, 'Fold', fold_i)
        feature_summary.insert(1, 'Task', f'Q{task}')
        
        feature_importance.append(feature_summary)
    
    selected_features.append(selected_features_)

selected_features_all = list(set(sum([x.tolist() for x in sum(selected_features, [])], [])))
selected_features_no = [x for x in all_features if x not in selected_features_all]
print('Number of feature selected: ', len(selected_features_all))

feature_if = pd.concat(feature_importance)
feature_if_mean = pd.DataFrame(feature_if.groupby('Feature')['Importance'].mean()).reset_index()
feature_if_mean['Feature domain'] = feature_if_mean['Feature'].apply(map_feat_to_group)
feature_if_mean.to_csv('feature_if.csv', index=False)

# 7.2 kurskal wallis with DSCF posthoc
rm_results = pg.kruskal(data=feature_if_mean, dv='Importance', between='Feature domain')
print()
print("Kurskal Wallis' test on feature importance")
print(rm_results)

# posthoc 
rm_posthoc = sp.posthoc_dscf(feature_if_mean, val_col='Importance', group_col='Feature domain')
print(rm_posthoc)

# 7.3 visualize the comparisons in a boxplot
# get signficance score
# combinations
numbers = list(range(3))
combinations = []
for dis in range(len(numbers)-1, 0, -1):
    for i in range(len(numbers) - dis):
        combinations.append((numbers[i], numbers[i + dis]))
significant_combinations =  [[combination, rm_posthoc.values[combination]] 
                              for combination in combinations]

# draw boxplot
fig, ax = plt.subplots(1, figsize=(5, 5))
bp = sns.boxplot(data=feature_if_mean, y='Importance', x='Feature domain', ax=ax, 
                  palette='Pastel1')
for patch in bp.artists:
    r, g, b, a = patch.get_facecolor()
    patch.set_facecolor((r, g, b, 0.76))

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# Get info about y-axis
bottom, top = ax.get_ylim()
yrange = top - bottom

# Significance bars
for i, significant_combination in enumerate(significant_combinations):
    # Columns corresponding to the datasets of interest
    x1 = significant_combination[0][0]
    x2 = significant_combination[0][1]
    # What level is this bar among the bars above the plot?
    level = len(significant_combinations) - i
    # Plot the bar
    bar_height = (yrange * 0.08 * level) + top
    bar_tips = bar_height - (yrange * 0.02)
    plt.plot(
        [x1, x1, x2, x2],
        [bar_tips, bar_height, bar_height, bar_tips], lw=1, c='k')
    # Significance level
    p = significant_combination[1]
    if p <= 0.001:
        sig_symbol = '***'
    elif p <= 0.01:
        sig_symbol = '**'
    elif p <= 0.05:
        sig_symbol = '*'
    else:
        sig_symbol = '/'  
    text_height = bar_height - (yrange * 0.01)
    plt.text((x1 + x2) * 0.5, text_height, sig_symbol, ha='center', c='k')

fig.savefig(out_dir + 'feature_importance.tif', dpi=300, bbox_inches='tight') 
fig.savefig(out_dir + 'feature_importance.png', dpi=300, bbox_inches='tight')

###############################################################################
# 8. task comparison ##########################################################
###############################################################################
# 8.1 get predictions per task
test_predict_probas = []
val_predict_probas = []

for fold_i, task_merge in tqdm(enumerate(task_merges), total=kfold, desc='task comparison'): 

    # train test val split 
    test_data = task_merge[task_merge['subset']=='test']
    val_data = task_merge[task_merge['subset']=='val']
    train_data = task_merge[task_merge['subset']=='train']
       
    test_predict_proba = {}
    val_predict_proba = {}

    selected_features_ = []
    for task in tasks:
        X_train = train_data[train_data['Task']==f'Q{task}'][all_features]
        y_train = train_data[train_data['Task']==f'Q{task}']['MF_Group']
        X_val = val_data[val_data['Task']==f'Q{task}'][all_features]
        y_val = val_data[val_data['Task']==f'Q{task}']['MF_Group']
        X_test = test_data[test_data['Task']==f'Q{task}'][all_features]
        y_test = test_data[test_data['Task']==f'Q{task}']['MF_Group']

        # set up a pipeline
        fold_k = fold_acc[fold_acc['Fold'] == fold_i+1]['K'].item()
        
        # set up a pipeline 
        model = Pipeline([
            # standard scaling
            ('scaler', StandardScaler()),       
            # impute nan with zero
            ('imputer', SimpleImputer(strategy='constant', fill_value=0)),     
            # remove correlated features 
            ('feature_removal_correlation', 
              DropCorrelatedFeatures(variables=None, method='pearson', threshold=0.8)),  
            # selection based on F value                     
            ('feature_selection', SelectKBest(f_classif, k=fold_k)),        
            # classifier
            ('classification', clf)                                      
        ])
        
        # train model to predict 
        model.fit(X_train, y_train)
        
        # the gold label is the same for all
        test_predict_proba[f'Q{task}'] = model.predict_proba(X_test)
        val_predict_proba[f'Q{task}'] = model.predict_proba(X_val)
    
    test_predict_probas.append(test_predict_proba)
    val_predict_probas.append(val_predict_proba)

y_true = test_data[test_data['Task']==f'Q{task}']['MF_Group'].tolist()

# 8.2 performance with each single task
task_single_prfa = []

for fold_index, fold_predict_proba in enumerate(test_predict_probas):
    fold_metrics = {}  

    for task in tasks:
        y_pred_proba = fold_predict_proba[f'Q{task}']
        y_pred = y_pred_proba.argmax(axis=1)
        
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        accuracy = accuracy_score(y_true, y_pred)
        
        fold_metrics[f'Q{task}'] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'acc': accuracy
        }
    
    task_single_prfa.append(fold_metrics)

# average across folds
task_single_prfa_df = pd.concat([pd.DataFrame(t).T for t in task_single_prfa])
task_single_res = task_single_prfa_df.groupby(task_single_prfa_df.index).mean().reset_index()
task_single_res.rename(columns={'index': 'Task'}, inplace=True)

# compare to the voting classifier
vote_prfa = fold_acc[fold_acc['Fold']=='Avg'][['test_precision', 'test_recall', 'test_f1', 'test_acc']].values
vote_df = pd.DataFrame(vote_prfa, columns=['precision', 'recall', 'f1', 'acc'])
vote_df.insert(0, 'Task', 'Vote') 
task_single_res = pd.concat([task_single_res, vote_df], ignore_index=True)

# performance differences = single_prfa - vote_prfa
task_single_res.loc[task_single_res['Task'] != 'Vote', ['precision', 'recall', 'f1', 'acc']] = \
    task_single_res.loc[task_single_res['Task'] != 'Vote', ['precision', 'recall', 'f1', 'acc']] - vote_prfa[0]


# 8.3 performance with ablation studies 
task_ablation_prfa = []

for fold_index, fold_predict_proba in enumerate(test_predict_probas):
    fold_metrics = {}  

    for task in tasks:
        y_pred_proba = [fold_predict_proba[f'Q{t}'] for t in tasks if t != task]
        y_pred = vote_result(y_pred_proba)
        
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        accuracy = accuracy_score(y_true, y_pred)
        
        fold_metrics[f'Q{task}'] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'acc': accuracy
        }
    
    task_ablation_prfa.append(fold_metrics)

# average across folds
task_ablation_prfa_df = pd.concat([pd.DataFrame(t).T for t in task_ablation_prfa])
task_ablation_res = task_ablation_prfa_df.groupby(task_ablation_prfa_df.index).mean().reset_index()
task_ablation_res.rename(columns={'index': 'Task'}, inplace=True)

# compare to the voting classifier
vote_prfa = fold_acc[fold_acc['Fold']=='Avg'][['test_precision', 'test_recall', 'test_f1', 'test_acc']].values
vote_df = pd.DataFrame(vote_prfa, columns=['precision', 'recall', 'f1', 'acc'])
vote_df.insert(0, 'Task', 'Vote') 
task_ablation_res = pd.concat([task_ablation_res, vote_df], ignore_index=True)

# performance differences = ablation_prfa - vote_prfa
task_ablation_res.loc[task_ablation_res['Task'] != 'Vote', ['precision', 'recall', 'f1', 'acc']] = \
    task_ablation_res.loc[task_ablation_res['Task'] != 'Vote', ['precision', 'recall', 'f1', 'acc']] - vote_prfa[0]

# 8.4 visualization 
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# single task
res_plot = task_single_res.melt(
    id_vars="Task", 
    value_vars=['precision', 'recall', 'f1', 'acc'], 
    var_name='Metric', 
    value_name='Perframance Decreases')
res_plot = res_plot[res_plot['Task'] != 'Vote']

# line plots
sns.lineplot(data=res_plot, x="Task", y="Perframance Decreases", hue="Metric", 
              marker='o', ax=ax1, palette='Pastel1')

# add dash line at 0
ax1.axhline(0, color='grey', linestyle='--')

# ax1.set_title('(A) Performance changes when using each individual task', fontsize=14)
ax1.set_ylabel('Performance Changes', fontsize=12)
ax1.tick_params(axis='both', labelsize=12)
ax1.set_xticklabels(['SelfInt', 'PastInt', 'PicDesp', 'Read', 'Recall'], fontsize=12)

# task ablation
res_plot = task_ablation_res.melt(
    id_vars="Task", 
    value_vars=['precision', 'recall', 'f1', 'acc'], 
    var_name='Metric', 
    value_name='Perframance Decreases')
res_plot = res_plot[res_plot['Task'] != 'Vote']

# line plots
sns.lineplot(data=res_plot, x="Task", y="Perframance Decreases", hue="Metric", 
              marker='o', ax=ax2, palette='Pastel1')

# add dash line at 0
ax2.axhline(0, color='grey', linestyle='--')

# ax2.set_title('(B) Performance changes when ablating each individual task', fontsize=14)
ax2.set_ylabel('Performance Changes', fontsize=12)
ax2.set_xticklabels(['SelfInt', 'PastInt', 'PicDesp', 'Read', 'Recall'], fontsize=12)
ax2.tick_params(axis='both', labelsize=12)

fig.savefig(out_dir + 'task_comparison_test.tif', dpi=300, bbox_inches='tight') 
fig.savefig(out_dir + 'task_comparison_test.png', dpi=300, bbox_inches='tight')

###############################################################################
# 9. task comparison for validation set - supplementary #######################
###############################################################################

y_true = val_data[val_data['Task']==f'Q{task}']['MF_Group'].tolist()

# 9.1 performance with each single task
task_single_prfa = []

for fold_index, fold_predict_proba in enumerate(val_predict_probas):
    fold_metrics = {}  

    for task in tasks:
        y_pred_proba = fold_predict_proba[f'Q{task}']
        y_pred = y_pred_proba.argmax(axis=1)
        
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        accuracy = accuracy_score(y_true, y_pred)
        
        fold_metrics[f'Q{task}'] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'acc': accuracy
        }
    
    task_single_prfa.append(fold_metrics)

# average across folds
task_single_prfa_df = pd.concat([pd.DataFrame(t).T for t in task_single_prfa])
task_single_res = task_single_prfa_df.groupby(task_single_prfa_df.index).mean().reset_index()
task_single_res.rename(columns={'index': 'Task'}, inplace=True)

# compare to the voting classifier
vote_prfa = fold_acc[fold_acc['Fold']=='Avg'][['val_precision', 'val_recall', 'val_f1', 'val_acc']].values
vote_df = pd.DataFrame(vote_prfa, columns=['precision', 'recall', 'f1', 'acc'])
vote_df.insert(0, 'Task', 'Vote') 
task_single_res = pd.concat([task_single_res, vote_df], ignore_index=True)

# performance differences = single_prfa - vote_prfa
task_single_res.loc[task_single_res['Task'] != 'Vote', ['precision', 'recall', 'f1', 'acc']] = \
    task_single_res.loc[task_single_res['Task'] != 'Vote', ['precision', 'recall', 'f1', 'acc']] - vote_prfa[0]


# 9.2 performance with ablation studies 
task_ablation_prfa = []

for fold_index, fold_predict_proba in enumerate(val_predict_probas):
    fold_metrics = {}  

    for task in tasks:
        y_pred_proba = [fold_predict_proba[f'Q{t}'] for t in tasks if t != task]
        y_pred = vote_result(y_pred_proba)
        
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        accuracy = accuracy_score(y_true, y_pred)
        
        fold_metrics[f'Q{task}'] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'acc': accuracy
        }
    
    task_ablation_prfa.append(fold_metrics)

# average across folds
task_ablation_prfa_df = pd.concat([pd.DataFrame(t).T for t in task_ablation_prfa])
task_ablation_res = task_ablation_prfa_df.groupby(task_ablation_prfa_df.index).mean().reset_index()
task_ablation_res.rename(columns={'index': 'Task'}, inplace=True)

# compare to the voting classifier
vote_prfa = fold_acc[fold_acc['Fold']=='Avg'][['val_precision', 'val_recall', 'val_f1', 'val_acc']].values
vote_df = pd.DataFrame(vote_prfa, columns=['precision', 'recall', 'f1', 'acc'])
vote_df.insert(0, 'Task', 'Vote') 
task_ablation_res = pd.concat([task_ablation_res, vote_df], ignore_index=True)

# performance differences = ablation_prfa - vote_prfa
task_ablation_res.loc[task_ablation_res['Task'] != 'Vote', ['precision', 'recall', 'f1', 'acc']] = \
    task_ablation_res.loc[task_ablation_res['Task'] != 'Vote', ['precision', 'recall', 'f1', 'acc']] - vote_prfa[0]

# 9.3 visualization 
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# single task
res_plot = task_single_res.melt(
    id_vars="Task", 
    value_vars=['precision', 'recall', 'f1', 'acc'], 
    var_name='Metric', 
    value_name='Perframance Decreases')
res_plot = res_plot[res_plot['Task'] != 'Vote']

# line plots
sns.lineplot(data=res_plot, x="Task", y="Perframance Decreases", hue="Metric", 
              marker='o', ax=ax1, palette='Pastel1')

# add dash line at 0
ax1.axhline(0, color='grey', linestyle='--')

ax1.set_title('(A) Performance changes when using each individual task', fontsize=14)
ax1.set_ylabel('Perframance Decreases', fontsize=12)
ax1.tick_params(axis='both', labelsize=12)
ax1.set_xticklabels(['SelfInt', 'PastInt', 'PicDesp', 'Read', 'Recall'], fontsize=12)

# task ablation
res_plot = task_ablation_res.melt(
    id_vars="Task", 
    value_vars=['precision', 'recall', 'f1', 'acc'], 
    var_name='Metric', 
    value_name='Perframance Decreases')
res_plot = res_plot[res_plot['Task'] != 'Vote']

# line plots
sns.lineplot(data=res_plot, x="Task", y="Perframance Decreases", hue="Metric", 
              marker='o', ax=ax2, palette='Pastel1')

# add dash line at 0
ax2.axhline(0, color='grey', linestyle='--')

ax2.set_title('(B) Performance changes when ablating each individual task', fontsize=14)
ax2.set_ylabel('Perframance Decreases', fontsize=12)
ax2.set_xticklabels(['SelfInt', 'PastInt', 'PicDesp', 'Read', 'Recall'], fontsize=12)
ax2.tick_params(axis='both', labelsize=12)


fig.savefig(out_dir + 'task_comparison_val.tif', dpi=300, bbox_inches='tight') 
fig.savefig(out_dir + 'task_comparison_val.png', dpi=300, bbox_inches='tight')

###############################################################################
# 10. Exploratory factor analysis  ############################################
###############################################################################

feature_matrix = StandardScaler().fit_transform(task_merge_df[selected_features_all].fillna(0).values)

# 10.1 PC-based parallel analysis 
# https://stackoverflow.com/a/70918704
pca = PCA(len(selected_features_all)-1)
pca.fit(feature_matrix)
transformedShapeMatrix = pca.transform(feature_matrix)

# random eigenvalues
np.random.seed(1234)
random_eigenvalues = np.zeros(len(selected_features_all)-1)
for i in trange(100, desc='parallel'):
    random_shapeMatrix = pd.DataFrame(np.random.normal(0, 1, list(feature_matrix.shape)))
    pca_random = PCA(len(selected_features_all)-1)
    pca_random.fit(random_shapeMatrix)
    transformedRandomShapeMatrix = pca_random.transform(random_shapeMatrix)
    random_eigenvalues = random_eigenvalues+pca_random.explained_variance_ratio_
random_eigenvalues = random_eigenvalues / 100

# visualize
fig, ax = plt.subplots(figsize=(5, 5))

ax.plot(pca.explained_variance_ratio_, '-o', color='#FFB996', label='Features')
ax.plot(random_eigenvalues, '-x', color='#9DBC98', label='Simulated data from parallel analysis')
ax.legend()
ax.set_title('Scree plot with parallel analysis')

fig.savefig(out_dir + 'EFA_parallel_analysis_scree_plot.png', dpi=300, bbox_inches='tight') 

# number of factors
num_factors = len(np.where(pca.explained_variance_ratio_ > random_eigenvalues)[0])

# 10.2 EFA 
fa = factor_analyzer.FactorAnalyzer(n_factors=num_factors, rotation='promax')
fa.fit(feature_matrix)

fa_loadings = fa.loadings_.copy()
fa_loadings[np.abs(fa_loadings) <= 0.3] = np.nan

fa_res = pd.DataFrame(fa_loadings, 
                      columns=[f'Factor_{x+1}' for x in range(num_factors)],
                      index=selected_features_all)
fa_res['Communalities'] = fa.get_communalities()
fa_res['Uniqueness'] = fa.get_uniquenesses()

fa_res.insert(0, 'Features', fa_res.index)
fa_res['Group'] = fa_res['Features'].map(map_feat_to_group)
fa_res = fa_res[['Features','Group'] + [f'Factor_{i+1}' for i in range(num_factors)] + ['Communalities', 'Uniqueness']]
fa_res.to_csv(out_dir + 'EFA.csv', index=False)


###############################################################################
# 11.Genralized estimating equation $##########################################
###############################################################################
# https://arxiv.org/pdf/2010.15869
# https://uvadoc.uva.es/bitstream/handle/10324/51995/Speech_Communication.pdf?sequence=1&isAllowed=y

gee_results = []
for dv_feat in tqdm(selected_features_all, desc='GEE'):
    
    df = task_merge_df.copy()
    df.years_edu = df.years_edu.fillna(11)
    
    if dv_feat not in acoustic_feats:
        df = df[df['Task'] != 'Q7rd']
    
    # transform
    df[dv_feat] = StandardScaler().fit_transform(df[dv_feat].values.reshape(-1, 1))
    
    # fit GEE
    formula = f'{dv_feat} ~ EdadEvaluacion + C(sexo) + years_edu + C(MF_Group, Treatment(0))'
    model = smf.gee(formula, 
                    data=df, 
                    groups=df['PAR'], 
                    time=df['Task'], 
                    family=sm.families.Gaussian(sm.families.links.Identity()), 
                    cov_struct=sm.cov_struct.Exchangeable())
    # fit the model 
    result = model.fit()
    # print(result.summary())

    # Deviance goodness-of-fit test
    deviance = result.deviance
    df_resid = result.df_resid
    fit_p = 1 - stats.chi2.cdf(deviance, df_resid)
    assert (fit_p > 0.05)
    
    # get results
    z = result.tvalues['C(MF_Group, Treatment(0))[T.1]']
    p = result.pvalues['C(MF_Group, Treatment(0))[T.1]']
    ci_l, ci_u = (result.conf_int().loc['C(MF_Group, Treatment(0))[T.1]', :] / result.bse['C(MF_Group, Treatment(0))[T.1]']).values.tolist()
     
    gee_results.append([dv_feat, z, p, ci_l, ci_u])

gee_results = pd.DataFrame(gee_results, columns=['Feat', 'Z', 'p', 'CI_lower', 'CI_upper'])

# FDR correction 
gee_results['Group'] = gee_results['Feat'].map(map_feat_to_group)
gee_results['q'] = multitest.fdrcorrection(gee_results['p'])[1]

# add more information
gee_results['Z_abs'] = gee_results['Z'].apply(abs)
gee_results = gee_results.sort_values(by='Z_abs', ascending=True).reset_index(drop=True) 
gee_results = pd.merge(gee_results, feature_if_mean, left_on='Feat', right_on='Feature', how='left')

# reorder
ordered_feats = task_merge_df.columns[2:322].tolist()
gee_results['order'] = gee_results['Feat'].apply(
    lambda x: ordered_feats.index(x) if x in ordered_feats else len(ordered_feats)
    )
gee_results = gee_results.sort_values(by='order').drop(columns='order').reset_index(drop=True)

gee_results.to_csv(out_dir + 'gee.csv', index=False)

# plot significant ones
gee_sig = gee_results[gee_results['q'] < 0.05]
fig, ax = plt.subplots(figsize=(9, 16))

# Plot error bar
ax.errorbar(gee_sig['Z'], range(len(gee_sig)), 
              xerr=[gee_sig['Z'] - gee_sig['CI_lower'], 
                    gee_sig['CI_upper'] - gee_sig['Z']], 
              fmt='o', capsize=5, ms=3, elinewidth=2, markeredgewidth=2,
              ecolor='#e18727ff', mfc='#7e6148ff', mec='#7e6148ff')

ax.set_yticks(range(len(gee_sig)))
ax.set_yticklabels(gee_sig['Feat'])
ax.set_xlabel('Z values')
ax.set_title('Z values with confidence interval', fontsize=12)
ax.axvline(0, color='gray', linestyle='dashed', linewidth=2)

fig.savefig(out_dir + 'group_compare_Z.tif', dpi=300, bbox_inches='tight') 
fig.savefig(out_dir + 'group_compare_Z.png', dpi=300, bbox_inches='tight') 



###############################################################################
# 12. Symptom correlations ####################################################
###############################################################################

feats_impt = gee_sig.Feat

print()

# panss item 
panss_score = {'Ausente':1, 'Muy leve':2, 'Leve':3, 
                'Moderado':4, 'Moderado-Severo':5, 
                'Severo':6, 'Extremadamente severo':7}
panss_items = ['PANSS_ItemP1', 'PANSS_ItemP2', 'PANSS_ItemP3', 
                'PANSS_ItemN1', 'PANSS_ItemN4', 'PANSS_ItemN6', 
                'PANSS_ItemG2','PANSS_ItemG5', 'PANSS_ItemG6','PANSS_ItemG9',]

# average
task_mean = demo[['codigoFamiliar', 'MF_Group', 'EdadEvaluacion', 'sexo', 'years_edu'] + panss_items].copy()
for panss_item in panss_items[:]:
    task_mean[panss_item] = task_mean[panss_item].apply(lambda x: panss_score[x]).astype(float)
    
for dv_feat in feats_impt:
    task_mean[dv_feat] = task_mean['codigoFamiliar'].map(
        dict(task_merge.groupby('PAR')[dv_feat].apply(lambda x: np.nanmean(x)))
        )
    
task_mean_sz = task_mean[task_mean['MF_Group']==1]
task_mean_sz['sexo'] = task_mean_sz['sexo'].apply(lambda x: 0 if x=='Male' else 1).astype(float)
task_mean_sz.years_edu = task_mean_sz.years_edu.fillna(11)

# correlation 
panss_corrs = pd.DataFrame()
for feat in tqdm(feats_impt, desc='panss_corr'):
    panss_corr = pg.pairwise_corr(task_mean_sz, 
                                  columns=[panss_items, [feat]], 
                                  method='spearman',
                                  padjust='fdr_bh') 
    panss_corr['q'] = multitest.fdrcorrection(panss_corr['p-unc'])[1]
    panss_corrs = pd.concat([panss_corrs, panss_corr])

# mask insignificant ones
panss_corrs_mask = panss_corrs.copy()
rs = panss_corrs['r'].values.copy()
rs[np.where(panss_corrs['q']>=0.05)] = 0
panss_corrs_mask['r'] = rs

# Pivot the DataFrame to create a matrix of r values
pivot_df = panss_corrs_mask.pivot(index='X', columns='Y', values='r')
        
# order the dataframe
pivot_df = pivot_df[feats_impt]
pivot_df = pivot_df.reindex(panss_items, axis=0)      
rps = np.full(pivot_df.shape, '').astype('U10')

# Pivot the DataFrame to create a matrix of p values
r_df = panss_corrs.pivot(index='X', columns='Y', values='r')

# order the dataframe
r_df = r_df[feats_impt]
r_df = r_df.reindex(panss_items, axis=0)

# Pivot the DataFrame to create a matrix of p values
q_df = panss_corrs.pivot(index='X', columns='Y', values='q')

# order the dataframe
q_df = q_df[feats_impt]
q_df = q_df.reindex(panss_items, axis=0)

# add annotation
rs = r_df.copy().values
qs = q_df.copy().values

rps = np.full(pivot_df.shape, '').astype('U10')
for i in range(pivot_df.shape[0]):
    for j in range(pivot_df.shape[1]):

        r = rs[i, j]
        q = qs[i, j]

        if q < 0.001:
            rp = f'{r:.3f}***'
        elif q < 0.010:
            rp = f'{r:.3f}**'
        elif q < 0.050:
            rp = f'{r:.3f}*'   
        else:
            rp = f'{r:.3f}'

        rps[i, j] = rp


pivot_df.index = [x.replace('PANSS', '').replace('_Item', '') for x in pivot_df.index]
rps = pd.DataFrame(rps, columns=pivot_df.columns, index=pivot_df.index)

# plot the heatmap
fig, ax = plt.subplots(figsize=(12, 40))
    
sns.heatmap(pivot_df.T, fmt="", annot=rps.T, annot_kws={"size": 13}, 
            cbar=True, 
            cbar_kws={"shrink": 0.5, "orientation": 'vertical'}, 
            vmin = -1, vmax = 1, 
            square=True, linewidth=.5, 
            # cmap=sns.diverging_palette(150, 240, s=80, l=50, n=9, center='light', as_cmap=True), 
            cmap='coolwarm',
            ax = ax
            )

ax.set(xlabel="", ylabel="")
ax.xaxis.tick_top()
ax.xaxis.set_tick_params(rotation=0, labelsize=13)
ax.yaxis.set_tick_params(rotation=0, labelsize=13)

fig.savefig(out_dir + 'panss_corr.tif', dpi=300, bbox_inches='tight') 
fig.savefig(out_dir + 'panss_corr.png', dpi=300, bbox_inches='tight') 
panss_corrs.to_csv(out_dir + 'panss_corr.csv', index=False) 
























