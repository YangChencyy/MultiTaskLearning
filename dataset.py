from typing import BinaryIO
import torch
import pandas as pd
from torch.utils.data import Dataset

def maketable(datapath,type):
    labelfilediectory = "drive/MyDrive/codes/MultiTaskLearning/Features/"+type+"_split.csv"
    datafilediectory = "drive/MyDrive/codes/MultiTaskLearning"+datapath
    labelset = pd.read_csv(labelfilediectory, sep=",")
    addlist = []
    for i in range(len(labelset['Participant_ID'])):
        dataroute = datafilediectory+str(labelset['Participant_ID'][i])+"_sentence_embedding.pt"
        try:
            _ = torch.load(dataroute)
            #addlist.append(str(labelset['Participant_ID'][i])+"_sentence_embedding.pt")
            addlist.append(dataroute)
        except:
            labelset = labelset.drop(labels = i, axis = 0)

    labelset['name'] = addlist
    return labelset

class MultiTaskDataset(Dataset):
    def __init__(self, df, mode = "both"):
        self.paths = list(df.name)
        self.Binary = list(df.PHQ_Binary) # or use df['label'] if needed
        self.Score = list(df.PHQ_Score)
        self.mode = mode

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx):
        # dealing with the text features
        text_feature = torch.load(self.paths[idx])
        # try:
        #     text_feature = torch.load(self.paths[idx])
        # except:
        #     text_feature = None
  
        # dealing with the audio features
        audio_feature = None

        #dealing with the labels
        #PHQ_Binary = torch.tensor(int(self.Binary[idx]), dtype=torch.int32)
        PHQ_Binary = torch.tensor(int(self.Binary[idx]), dtype=torch.int64)
        PHQ_Score = torch.tensor(int(self.Score[idx]), dtype=torch.int64)
        
        if self.mode == "text":
            return text_feature, (PHQ_Binary, PHQ_Score)
        elif self.mode == "audio":
            return audio_feature, (PHQ_Binary, PHQ_Score)
        return (text_feature, audio_feature), (PHQ_Binary, PHQ_Score)   # how to combine two features



