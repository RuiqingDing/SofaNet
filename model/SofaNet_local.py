import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Function

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class SofaNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=2, num_layers=1, return_embedding=True):
        super(SofaNet, self).__init__() 
        self.input_dim = input_dim # 输入的特征的个数
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.return_embedding = return_embedding

        # Define the GRU layer
        self.gru1 = nn.GRU(self.input_dim*2, self.hidden_dim, self.num_layers, dropout=0.1, batch_first=True)
        self.gru2 = nn.GRU(self.input_dim*2, self.hidden_dim, self.num_layers, dropout=0.1, batch_first=True)
        self.gru3 = nn.GRU(self.input_dim*2, self.hidden_dim, self.num_layers, dropout=0.1, batch_first=True)
        self.gru4 = nn.GRU(self.input_dim*2, self.hidden_dim, self.num_layers, dropout=0.1, batch_first=True)
        
        # Define the output layer
        self.linear1 = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim // 2, 3)
        )
        self.linear2 = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim // 2, 3)
        )
        self.linear3 = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim // 2, 3)
        )
        self.linear4 = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim // 2, 2)
        )
        # self.bn = nn.BatchNorm1d(self.hidden_dim*4+3+3+3+2)
        self.linear = nn.Sequential(
            nn.Linear(self.hidden_dim*4+3+3+3+2, self.hidden_dim*4 // 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim*4 // 2, output_dim)
        )

    def forward(self,x):
        x_delta = x[:, 1:6, :] - x[:, 0:5, :]
        x_delta0 = torch.zeros(x.size(0), 1, x.size(2)).to(DEVICE)
        x_delta = torch.cat((x_delta0, x_delta), 1)
        x_concat = torch.cat((x, x_delta), 2)

        gru_out1, hidden1 = self.gru1(x_concat) 
        output1 = self.linear1(gru_out1.permute(1,0,2)[-1]) 
        gru_out2, hidden2 = self.gru2(x_concat) 
        output2 = self.linear2(gru_out2.permute(1,0,2)[-1])
        gru_out3, hidden3 = self.gru3(x_concat) 
        output3 = self.linear3(gru_out3.permute(1,0,2)[-1])
        gru_out4, hidden4 = self.gru4(x_concat)
        output4 = self.linear4(gru_out4.permute(1,0,2)[-1])
        gru_out = torch.cat((gru_out1.permute(1,0,2)[-1], gru_out2.permute(1,0,2)[-1], gru_out3.permute(1,0,2)[-1], gru_out4.permute(1,0,2)[-1], output1, output2, output3, output4), 1)

        output = self.linear(gru_out)
        if self.return_embedding:
            return [output1, output2, output3, output4, gru_out, output]
        else:
            return [output1, output2, output3, output4, output]

