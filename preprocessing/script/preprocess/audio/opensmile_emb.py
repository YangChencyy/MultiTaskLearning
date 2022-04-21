import os
import time

import numpy as np
import pandas as pd

# import audb
import audiofile
import opensmile

dir = "audios/"
transcript_dir = "Transcript/{}_Transcript.csv"
files = os.listdir(dir)

Opensmile_dir = "Opensmile_eGeMAPSv02/"
existing_embs = os.listdir(Opensmile_dir)

for filename in files:
    print(filename)
    if "{}_eGeMAPS.npy".format(filename[:3]) in existing_embs:
        continue
    df = pd.read_csv(transcript_dir.format(filename[:3]))
    vec = np.empty((df.shape[0], 88))
    count = 0
    for _, row in df.iterrows():
        signal, sampling_rate  =audiofile.read(
            dir+filename,
            offset=row["Start_Time"],
            duration=row["End_Time"]-row["Start_Time"],
        )

        smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.Functionals,
        )

        vec[count] = smile.process_signal(
            signal,
            sampling_rate
        )
        count += 1
    print(vec.shape)
    with open("Opensmile_eGeMAPSv02/{}_eGeMAPS.npy".format(filename[:3]), 'wb') as f:
        np.save(f, vec)



