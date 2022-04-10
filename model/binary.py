import torch
import torch.nn as nn
import warnings
warnings.filterwarnings("ignore")



class BinaryModel(nn.Module):
    def __init__(self, ps=0.5):
        #super(MultiTaskModel,self).__init__()
        super().__init__()

        self.fc1=nn.Linear(472,256)
        # self.fc3=nn.Linear(1024,2048)
        # self.fc4=nn.Linear(2048,512)
        self.fc5=nn.Linear(256,64)
        self.relu1=nn.ReLU()
        # self.drop1=nn.Dropout(p=ps)
        self.fcL1=nn.Linear(64,2)
        self.sigmoid=nn.Sigmoid()


    def forward(self, x):

        x = torch.flatten(x, start_dim = 1)
        x.requires_grad=True
        PHQ_B1 = self.fc1(x)
        PHQ_B2 = self.relu1(PHQ_B1)


        PHQ_B3 = self.fc5(PHQ_B2)
        PHQ_B3 = self.relu1(PHQ_B3)
        #print(PHQ_B3.shape)
        PHQ_B4 = self.fcL1(PHQ_B3)

        #print(PHQ_B4.shape)
        PHQ_Binary = self.sigmoid(PHQ_B4)

    
        return PHQ_Binary