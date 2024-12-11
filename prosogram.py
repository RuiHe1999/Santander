# 1. packages
import os
import subprocess
import shutil
import glob
import pandas as pd
from tqdm import tqdm

# 2. functions
def extract_profile(wav_path):
    """
    

    Args:
        wav_path (str): pathway to the speech file.

    Returns:
        profile (DataFrame) : the result of prosogram.

    """
    
    # get directory
    original_cwd = os.getcwd()
    script_dir = os.path.dirname(os.path.abspath("__file__"))

    # create a filefoder to store temporary file
    tmp_dir = os.path.abspath("./tmp")
    shutil.rmtree(tmp_dir, ignore_errors=True)
    os.makedirs(tmp_dir, exist_ok=True)

    try:
        # copy all prosogram scripts into the temporary folder 
        for p in glob.glob(os.path.join(script_dir, "*.praat")):
            shutil.copyfile(p, os.path.join(tmp_dir, os.path.basename(p)))

        shutil.copyfile("Praat.exe", os.path.join(tmp_dir, "Praat.exe"))

        # copy the speech file into the temporary folder
        wav_copy_path = os.path.join(tmp_dir, os.path.basename(wav_path))
        shutil.copyfile(wav_path, wav_copy_path)

        # create a praat work file in the temporary folder with setting

        # use default setting here
        # if needed, add custome settings in prosogram_variants of job_content
        # for example, if the custom settings are: g=0.32 dg=20 dmin=0.035
        # the job_content should be:
        # """
        # include prosomain.praat
        # @prosogram_variants: "file={os.path.abspath(wav_copy_path)} save=yes draw=no 
        # g=0.32 dg=20 dmin=0.035"
        # exit
        # """

        job_contents = f"""include prosomain.praat
        @prosogram: "file={os.path.abspath(wav_copy_path)} save=yes draw=no"
        exit"""

        job_path = os.path.join(tmp_dir, "job.praat")
        with open(job_path, "w") as job_file:
            job_file.write(job_contents)

        os.chdir(tmp_dir)

        # create a command line to run praat
        invocation = f"Praat.exe --run {os.path.abspath(job_path)}"
        status, output = subprocess.getstatusoutput(invocation)
        # Give a warning when the processing failed (status is not 0)
        if status != 0:
            print(output)
            raise Warning("FAILED: Praat failed! The outputs are printed above.")

        # Read the result file 
        profile_path = os.path.join(tmp_dir, 
                                    wav_copy_path.replace(".wav", "") + \
                                        "_profile_data.txt")
        profile = pd.read_csv(profile_path, sep="\t")
    
    finally:
        # delete all  contents in the temporary filefolder
        shutil.rmtree(tmp_dir, ignore_errors=True)
        os.chdir(original_cwd)
    
    return profile 


# 4. commands
path = 'E:/A-Horace/PhD/202404-Santander/santander/Audios'
wav_paths = glob.glob(os.path.join(path, "**/*.wav"), recursive=True) 

# get all data within one data frame
profile = pd.DataFrame()
for wav_path in tqdm(wav_paths):
    profile = pd.concat([profile,
                          extract_profile(wav_path)])

# drop variables
drop_vars = ['NrofNuclei', 
'PropPhon', 
'F0MedianHz', 
'F0MeanHz', 
'F0MeanInST', 
'F0StdevHz', 
'PitchTopHz', 
'PitchBottomHz', 
'RawF0_p02', 
'RawF0_p98', 
'nPVI_voweldur', 
'nPVI_sylldur', ]

# add the identifier column
profile['Audios'] = wav_paths
profile = profile.reindex(columns=['Audios']+profile.columns.tolist()[1:])
profile_drop = profile.drop(columns=[drop_vars])

# # save as a csv file
# profile.to_csv('Prosogram_results.csv', encoding='utf-8-sig', index=False)



