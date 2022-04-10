import torch
import torch.nn as nn
import warnings
warnings.filterwarnings("ignore")



class MultiTaskModel(nn.Module):
    def __init__(self, ps=0.5):
        #super(MultiTaskModel,self).__init__()
        super().__init__()
        #self.encoder = create_body(arch)
        #self.fc1 = create_head(1024,1,ps=ps)
        #self.fc2 = create_head(1024,1,ps=ps)
        # self.fc1 = create_head(1,1,ps=ps)
        # self.fc2 = create_head(1,25,ps=ps)
        self.fc1=nn.Linear(768,512)
        self.relu1=nn.ReLU()
        self.drop1=nn.Dropout(p=0.25)
        self.fcL1=nn.Linear(512,2)
        self.sigmoid=nn.Sigmoid()
        self.fc2=nn.Linear(768,512)
        self.relu2=nn.ReLU()
        self.drop2=nn.Dropout(p=0.25)
        self.fcL2=nn.Linear(512,25)

    def forward(self, x):

        #x = self.encoder(x)
        #print(x.shape)
        #x=torch.squeeze(x)
        x = torch.flatten(x, start_dim = 1)
        x.requires_grad=True
        # print(x.requires_grad)
        # print(x.shape)
        PHQ_B1 = self.fc1(x)
        #print(PHQ_B1.shape)
        PHQ_B2 = self.relu1(PHQ_B1)

        #print(PHQ_B2.shape)
        PHQ_B3 = self.drop1(PHQ_B2)

        #print(PHQ_B3.shape)
        PHQ_B4 = self.fcL1(PHQ_B3)

        #print(PHQ_B4.shape)
        PHQ_Binary = self.sigmoid(PHQ_B4)
        
        PHQ_Score = self.fc2(x)    # think about how to change the architeture

        PHQ_Score = self.relu2(PHQ_Score)
 
        PHQ_Score = self.drop2(PHQ_Score)

        PHQ_Score = self.fcL2(PHQ_Score)
        # PHQ_Binary.requires_grad=True
        # PHQ_Score.requires_grad=True
        return [PHQ_Binary, PHQ_Score]


