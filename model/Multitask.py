import torch
import torch.nn as nn

from fastai import *
from fastai.vision import create_body, create_head
from fastai.layers import MSELossFlat, CrossEntropyFlat
from torchvision import transforms
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

class MultiTaskLossWrapper(nn.Module):
    def __init__(self, task_num):
        #super(MultiTaskLossWrapper, self).__init__()
        super().__init__()
        self.task_num = task_num
        self.log_vars = nn.Parameter(torch.zeros((task_num),requires_grad=True))

    def forward(self, preds, PHQ_Binary, PHQ_Score):
        # mse=MSELossFlat()
        crossEntropy = nn.CrossEntropyLoss()
        print(PHQ_Binary.requires_grad)
        print(PHQ_Score.requires_grad)
        crossEntropy.requires_grad = True
        # preds[0].requires_grad_ = True
        PHQ_Binary = PHQ_Binary.float()
        PHQ_Binary.requires_grad = True
        # preds[1].requires_grad_ = True
        PHQ_Score = PHQ_Score.float()
        PHQ_Score.requires_grad = True
        print(preds[0], PHQ_Binary)
        # with torch.no_grad():
        loss0=crossEntropy(preds[0], PHQ_Binary.long())
        # loss0=crossEntropy(torch.argmax(preds[0], dim=1),PHQ_Binary)
        #loss0 = crossEntropy(preds[0], PHQ_Binary)
        print("ccc")
        print(loss0.requires_grad)    
        loss1 = crossEntropy(preds[1], PHQ_Score.long())
        # loss0=loss0.long()
        # loss1=loss1.long()
        print(loss1.requires_grad)
        precision0 = torch.exp(-self.log_vars[0])
        loss0 = precision0*loss0 + self.log_vars[0]

        precision1 = torch.exp(-self.log_vars[1])
        loss1 = precision1*loss1 + self.log_vars[1]
        print("sss")
        print(loss0.requires_grad)
        print(loss1.requires_grad)

        loss = loss0 + loss1
        print(loss.requires_grad)
        
        return loss.long()
        
def criterion(y_pred, y_true, log_vars):
  loss = 0
  for i in range(len(y_pred)):
    precision = torch.exp(-log_vars[i])
    diff = (y_pred[i]-y_true[i])**2.
    loss += torch.sum(precision * diff + log_vars[i], -1)
  return torch.mean(loss)

def acc_Binary(preds, PHQ_binary, PHQ_Score): return root_mean_squared_error(preds[0], PHQ_binary)
def acc_Score(preds, PHQ_binary, PHQ_Score): return accuracy(preds[1], PHQ_Score)

def metrics():
    return [acc_Binary, acc_Score]




