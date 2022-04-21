import numpy as np
import pandas as pd
import os


transcript_dir = "Transcript/{}_Transcript.csv"
files = os.listdir("Transcript/")
index_path = "Transcript/index/{}_index.npy"
bert_dir = "BERT/sentence_embedding/{}_sentence_embedding.npy"

stop_sentence_list = [
    "thank you"
]
for filename in files:
    if filename[-4:] != ".csv":
        continue
    print(filename)
    df = pd.read_csv(transcript_dir.format(filename[:3]))
    text_index_list = []
    for id, row in df.iterrows():
        tokens = row['Text'].split()
        if len(tokens) < 2:
            continue
        if row['Text'] in stop_sentence_list:
            continue
        text_index_list.append(id)
    
    print(df.shape[0], len(text_index_list))

    
    vec = np.array(text_index_list)
    with open(index_path.format(filename[:3]), 'wb') as f:
        np.save(f, vec)
    print(vec.shape)