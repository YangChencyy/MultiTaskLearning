import torch
import pandas as pd
from torch.utils.data import Dataset
import numpy as np

def maketable(local_path,type):
    text_PATH = local_path + "/Features/BERT/sentence_embeddingnew/"
    audio_PATH = local_path + "/Features/Audio/Opensmile_eGeMAPSv02/"
    labelfilediectory = local_path + "/Features/" + type + "_split.csv"
    labelset = pd.read_csv(labelfilediectory, sep=",")
    data = []
    names = ["Participant_ID", "Utterance_ID", "text_PATH", "audio_PATH", "PHQ_Binary", "PHQ_Score"]
    for i in range(len(labelset['Participant_ID'])):
        if labelset['PHQ_Score'][i] > 15 or labelset['PHQ_Score'][i] < 1:
          textdata_PATH = text_PATH + str(labelset['Participant_ID'][i]) + "_sentence_embedding.npy"
          audiodata_PATH = audio_PATH + str(labelset['Participant_ID'][i]) + "_eGeMAPS.npy"
          binary = labelset['PHQ_Binary'][i]
          score = labelset['PHQ_Score'][i]
          pID = labelset['Participant_ID'][i]
          # print(textdata_PATH)
          try:
              t = np.load(textdata_PATH)
              d = np.load(audiodata_PATH)
              for j in range(len(t)):
                  data.append([pID, j, textdata_PATH, audiodata_PATH, binary, score])
              # print("try")
          except:
              pass

    result_pd = pd.DataFrame(data, columns = names)
    return result_pd

class MultiTaskDataset(Dataset):
    def __init__(self, df, mode = "both"):
        self.textpaths = list(df.text_PATH)
        self.audiopaths = list(df.audio_PATH)
        self.utternace= list(df.Utterance_ID)
        self.Binary = list(df.PHQ_Binary) # or use df['label'] if needed
        self.Score = list(df.PHQ_Score)
        self.mode = mode

    def __len__(self): return len(self.textpaths)

    def __getitem__(self, idx):
        # dealing with the text features
        text_featurebignp = np.load(self.textpaths[idx])
        text_featurebig= torch.tensor(text_featurebignp, dtype = torch.float)
        audio_featurebignp = np.load(self.audiopaths[idx])
        audio_featurebig= torch.tensor(audio_featurebignp, dtype = torch.float)
        text_feature = text_featurebig[self.utternace[idx]].reshape(1,-1)
        audio_feature = audio_featurebig[self.utternace[idx]].reshape(1,-1)

        # try:
        #     text_feature = torch.load(self.paths[idx])
        # except:
        #     text_feature = None
  
        # dealing with the audio features
        #audio_feature = None

        #dealing with the labels
        #PHQ_Binary = torch.tensor(int(self.Binary[idx]), dtype=torch.int32)
        PHQ_Binary = torch.tensor(int(self.Binary[idx]), dtype=torch.int64)
        PHQ_Score = torch.tensor(int(self.Score[idx]), dtype=torch.int64)
        dataembedding= torch.cat((text_feature,audio_feature),dim=1)
        if self.mode == "text":
            return text_feature, (PHQ_Binary, PHQ_Score)
        elif self.mode == "audio":
            return audio_feature, (PHQ_Binary, PHQ_Score)
        return dataembedding, (PHQ_Binary, PHQ_Score)   # how to combine two features

def get_class_stats(train_data):
    """
    Calculate the number of utterances for each emotion label.
    Input:
        - train_data: training data on which to get statistics.
        - label_index: indices for labels
    Return:
        - output: a list of length len(label_index) (in this case 7), that stores
            the number of utterances for each emotion label. Results should be
            in the order specified by label_index.
    """
    
    output1 = [0, 0]
    output2 = [0] * 24
    for i in train_data["PHQ_Binary"]:
        output1[i] += 1
    for i in train_data["PHQ_Score"]:
        output2[i] += 1
    return torch.tensor(output1, dtype=torch.float32), torch.tensor(output2, dtype=torch.float32)



def get_class_stats_binary(train_data):
    """
    Calculate the number of utterances for each emotion label.
    Input:
        - train_data: training data on which to get statistics.
        - label_index: indices for labels
    Return:
        - output: a list of length len(label_index) (in this case 7), that stores
            the number of utterances for each emotion label. Results should be
            in the order specified by label_index.
    """
    
    output1 = [0, 0]
    for i in train_data["PHQ_Binary"]:
        output1[i] += 1
    return torch.tensor(output1, dtype=torch.float32)






    