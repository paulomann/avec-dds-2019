import torch
import torch.nn as nn
from avec.data import OpenFaceDatasetLSTM
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F


class LSTM(nn.Module):

    def __init__(self, input_size, hidden_size):
        super(LSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(self.input_size, hidden_size)
        self.fc = nn.Linear(self.hidden_size, 1)
        self.init_weights()

    def forward(self, x, lengths):
        max_len = lengths.max().item()
        batch_size = x.size(0)
        x = x.permute(1,0,2)[0:max_len]
        x = pack_padded_sequence(x, lengths)
        lstm_out, (ho, _) = self.lstm(x)
        # lstm_out, lengths = pad_packed_sequence(lstm_out)
        # avg_pool = F.adaptive_avg_pool1d(
        #     lstm_out.permute(1,2,0),1).view(batch_size,-1)
        # max_pool = F.adaptive_max_pool1d(
        #     lstm_out.permute(1,2,0),1).view(batch_size,-1)
        # x = torch.cat([ho[-1], avg_pool, max_pool], dim=1)
        x = ho[-1]
        x = self.fc(x)
        return x.squeeze()
    
    def init_weights(self):
        """
        Here we reproduce Keras default initialization weights to 
        initialize Embeddings/LSTM weights
        """
        ih = (param.data for name, param in self.named_parameters()
              if 'weight_ih' in name)
        hh = (param.data for name, param in self.named_parameters()
              if 'weight_hh' in name)
        b = (param.data for name, param in self.named_parameters()
             if 'bias' in name)
        for t in ih:
            nn.init.xavier_uniform_(t)
        for t in hh:
            nn.init.orthogonal_(t)
        for t in b:
            nn.init.constant_(t, 0)