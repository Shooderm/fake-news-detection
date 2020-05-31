import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTM_model(nn.Module):
    def __init__(self):
        super(LSTM_model, self).__init__()
        self.lstm = nn.LSTM(emb_size, hidden, batch_first=True, dropout=0.2)
        self.dropout = nn.Dropout(0.2)
        self.hidden = nn.Linear(hidden, hidden)
        self.out  = nn.Linear(hidden, num_classes)

    def forward(self, x):
        lstm_out,(x,c_n) = self.lstm(x)
        x = self.dropout(lstm_out[:,-1,:])
        x = self.out(x)
        x = F.log_softmax(x, dim=-1)
        return x
