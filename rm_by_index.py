import numpy as np
import os


######################################
# EDIT HERE
index_path = "drive/My Drive/MT/Features/index/{}_index.npy"

text_embedding_path = "drive/My Drive/MT/Features/BERT/sentence_embeddingnew/"

audio_embedding_path = "drive/My Drive/MT/Features/Audio/Opensmile_eGeMAPSv02/{}_eGeMAPS.npy"

#Change here if you do not want to overwrite on the previous data
text_embedding_output_path = text_embedding_path

audio_embedding_output_path = audio_embedding_path

#text_embedding_output_path = "drive/My Drive/coding/MultiTaskLearning/MultiTaskLearning/Features/BERT/rm_stopword/"

#audio_embedding_output_path = "drive/My Drive/coding/MultiTaskLearning/MultiTaskLearning/Features/Audio/Opensmile_eGeMAPSv02/rm_stopword/{}_eGeMAPS.npy"
# STOP EDITING
######################################

# Process text embedding first
def rmfunction():
  for filename in os.listdir(text_embedding_path):
    with open(index_path.format(filename[:3]), 'rb') as f:
        index = np.load(f)
    
    with open(text_embedding_path + filename, 'rb') as f:
        text_emb = np.load(f)

    with open(audio_embedding_path.format(filename[:3]), 'rb') as f:
        audio_emb = np.load(f)


    text_emb = text_emb[index]
    audio_emb = audio_emb[index]

    with open(text_embedding_output_path + filename, 'wb') as f:
        np.save(f, text_emb)

    with open(audio_embedding_output_path.format(filename[:3]), 'wb') as f:
        np.save(f, audio_emb)

