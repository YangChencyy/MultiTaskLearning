import torch
import torch.nn as nn
from torch.nn import init
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence

class TextCNNEncoder(nn.Module):
    """Text encoder using CNN"""
    def __init__(self, kernel_size, num_channel):
        """
        Input:
            - kernel_size: a list for size of the kernels. e.g. [3, 4, 5] means we
                will have three kernels with size 3, 4, and 5 respectively.
            - num_channel: number of output channels for each kernel.

        A few key steps of the network:
            conv -> relu -> global max pooling -> concatenate
        
        Here we construct a list of 1d convolutional networks and store them in one pytorch object
        called ModuleList. Note we have varying kernel size and padding over this list, and
        later in the forward function we can iterate over self.convs to pass data through each network
        we've just set up.
        """
        super().__init__()
        self.convs = nn.ModuleList([nn.Conv1d(384, num_channel, k,
            padding=k // 2) for k in kernel_size])
    
    def forward(self, text_emb):
        """
        Input:
            - text_emb: input utterances with shape (N, L, 300), where N is the
                number of utterances in a batch, L is the longest utterance.
                Note we concatenate utterances from all dialogues.
        Return:
            - output: encoded utterances with shape (N, len(kernel_size) * num_channel)

        The purpose of a forward function is exactly what it sounds like, here we tell
        pytorch how to pass data "forward" through our network. Pytorch will automatically
        calculate a backward function based off of this forward function.
        """

        output = None
        
        # TextCNN forward
        output = torch.transpose(text_emb, 1, 2)
        # Loop through each convolutional layer, passing our data in and then through a relu activation function
        output = [F.relu(conv(output)) for conv in self.convs]
        # Perform a max pooling over each convolutional output
        output = [i.max(dim=2)[0] for i in output]
        output = torch.cat(output, 1)

        return output
class TextCNNEncoder2(nn.Module):
    """Text encoder using CNN"""
    def __init__(self, kernel_size, num_channel):
        """
        Input:
            - kernel_size: a list for size of the kernels. e.g. [3, 4, 5] means we
                will have three kernels with size 3, 4, and 5 respectively.
            - num_channel: number of output channels for each kernel.

        A few key steps of the network:
            conv -> relu -> global max pooling -> concatenate
        
        Here we construct a list of 1d convolutional networks and store them in one pytorch object
        called ModuleList. Note we have varying kernel size and padding over this list, and
        later in the forward function we can iterate over self.convs to pass data through each network
        we've just set up.
        """
        super().__init__()
        self.convs = nn.ModuleList([nn.Conv1d(88, num_channel, k,
            padding=k // 2) for k in kernel_size])
    
    def forward(self, text_emb):
        """
        Input:
            - text_emb: input utterances with shape (N, L, 300), where N is the
                number of utterances in a batch, L is the longest utterance.
                Note we concatenate utterances from all dialogues.
        Return:
            - output: encoded utterances with shape (N, len(kernel_size) * num_channel)

        The purpose of a forward function is exactly what it sounds like, here we tell
        pytorch how to pass data "forward" through our network. Pytorch will automatically
        calculate a backward function based off of this forward function.
        """

        output = None
        
        # TextCNN forward
        output = torch.transpose(text_emb, 1, 2)
        # Loop through each convolutional layer, passing our data in and then through a relu activation function
        output = [F.relu(conv(output)) for conv in self.convs]
        # Perform a max pooling over each convolutional output
        output = [i.max(dim=2)[0] for i in output]
        output = torch.cat(output, 1)

        return output


class ScoreModel(nn.Module):
    def __init__(self, kernel_size, num_channel,ps=0.5):
        #super(MultiTaskModel,self).__init__()
        super().__init__()
        self.textcnn=TextCNNEncoder(kernel_size=kernel_size,num_channel=num_channel)
        self.textcnn2=TextCNNEncoder2(kernel_size=kernel_size,num_channel=num_channel)
        cnn_out_dim=len(kernel_size)*num_channel*2
        self.fc1=nn.Linear(472,256)
        # self.fc3=nn.Linear(1024,2048)
        # self.fc4=nn.Linear(2048,512)
        self.fc5=nn.Linear(256,64)
        self.relu1=nn.ReLU()
        # self.drop1=nn.Dropout(p=ps)
        self.fcL1=nn.Linear(64,2)
        self.drop=nn.Dropout(p=ps)
        self.linear1=nn.Linear(cnn_out_dim,64)
        self.linear2=nn.Linear(len(kernel_size)*num_channel,64)
        self.linear3=nn.Linear(988,8)
        self.fcLd=nn.Linear(len(kernel_size)*num_channel,2)
        #self.linear2=nn.Linear(64,2)
        self.sigmoid=nn.Sigmoid()
        self.softmax=nn.Softmax()


    def forward(self, text_emb,audio_emb):
        text_emb = self.textcnn(text_emb)
        #audio_emb = self.textcnn2(audio_emb)
        print(text_emb.shape)
        
        #x = torch.cat((text_emb, audio_emb), dim=1)
        #x = self.drop(x)
        #x = self.linear1(x)
        #print(audio_emb.shape)
        audio= torch.sum(audio_emb,dim=1)/10
        x = torch.cat((text_emb, audio), dim=1)
        #print(audio.shape)
        x = self.drop(x)
        x = self.linear3(x)
        # x = self.relu1(x)
        # x = self.fcL1(x)
        #x = self.fcLd(audio_emb)
        # PHQ_Binary= self.sigmoid(x)
        #PHQ_Score= self.softmax(x)
        # x = torch.flatten(x, start_dim = 1)
        # x.requires_grad=True
        # PHQ_B1 = self.fc1(x)
        # PHQ_B2 = self.relu1(PHQ_B1)


        # PHQ_B3 = self.fc5(PHQ_B2)
        # PHQ_B3 = self.relu1(PHQ_B3)
        # #print(PHQ_B3.shape)
        # PHQ_B4 = self.fcL1(PHQ_B3)

        # #print(PHQ_B4.shape)
        # PHQ_Binary = self.sigmoid(PHQ_B4)

        return x
        #return PHQ_Score
        #return PHQ_Binary