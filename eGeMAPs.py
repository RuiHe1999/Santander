import os
import glob
import opensmile
import pandas as pd
from tqdm import tqdm 

# set up feature extractor
smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.Functionals,
)

# all wav files
path = 'Audios/'
wav_paths = glob.glob(os.path.join(path, "**/*.wav"), recursive=True) 

# extract features
results = pd.DataFrame()
for filename in tqdm(wav_paths):
    result = smile.process_file(filename)
    result.insert(0, 'PAR', filename.split('/')[1])
    result.insert(1, 'Task', filename.split('/')[-1].split('.')[0])
    results = pd.concat([results, result])

results.to_csv('Features/eGeMAPs.csv', index=False)






















