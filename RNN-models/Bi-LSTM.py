import torch
import torch.nn as nn
import torch.nn.functional as F

class BiLSTM_model(nn.Module):
    def __init__(self):
        super(BiLSTM_model, self).__init__()
        self.bilstm = nn.LSTM(emDataLoadeb_size, hidden,batch_first=True, num_layers=2, 
                              bidirectional=True, dropout=0.6)
        self.dropout = nn.Dropout(0.6)
        self.relu = nn.LeakyReLU(negative_slope=0.01,inplace=True)
        self.hidden = nn.Linear(d2v_size, 64)
        if emb_type==4 or emb_type==5 or emb_type==6:
            self.out  = nn.Linear(hidden*2++d2v_size, num_classes)
        elif emb_type==2:
            self.out = nn.Linear(64, num_classes)
        else:
            self.out = nn.Linear(hidden*2, num_classes)
        
    def forward(self, input, d2v):
        if emb_type!=2:      
            lstm_out,(h_n,c_n) = self.bilstm(input)
            x = torch.cat((h_n[0,:,:],h_n[1,:,:]),1)
        else:
            x = d2v
            x = self.hidden(x)
        if emb_type==4 or emb_type==5 or emb_type==6:
            x = torch.cat((x,d2v), 1)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.out(x)
        x = F.log_softmax(x, dim=-1)
        return x
