from sentence_transformers import SentenceTransformer, LoggingHandler
import numpy as np
import logging
import pandas as pd
import os

model = SentenceTransformer('all-MiniLM-L6-v2')

transcript_dir = "Transcript/{}_Transcript.csv"
files = os.listdir("Transcript/")

output_path = "BERT/sentence_embedding/{}_sentence_embedding.npy"
for filename in files:
    print(filename)
    df = pd.read_csv(transcript_dir.format(filename[:3]))
    text_list = []
    for _, row in df.iterrows():
        text_list.append(row['Text'])
        vec = model.encode(text_list)
    
    vec = np.array(vec)
    with open(output_path.format(filename[:3]), 'wb') as f:
        np.save(f, vec)
    print(vec.shape)
    
    




