from operator import length_hint
import pandas as pd
import os
import numpy as np
# import csv

MFCC_path = "OpenSMILE2.3.0_mfcc/"
save_path = "wav_emb/"

mfcc_type = ["pcm_fftMag_mfcc[{}]", "pcm_fftMag_mfcc_de[{}]", "pcm_fftMag_mfcc_de_de[{}]"]
headers = []

for mfcc in mfcc_type:
    for i in range(13):
        headers.append(mfcc.format(str(i)))

length_list = []
save_path_list = os.listdir(save_path)
for path in os.listdir(MFCC_path):
    print(path)
    if path[:-4] + '.npy' not in save_path_list:
        df = pd.read_csv(MFCC_path + path, sep=";")
        vec = df[headers].to_numpy()
        with open(save_path + path[:-4] + '.npy', 'wb') as f:
            np.save(f, vec)
    length_list.append(vec.shape[0])
        
print(sum(length_list) / len(length_list))
