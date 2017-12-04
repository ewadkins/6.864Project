import torch.nn as nn
import torch.nn.functional as F


class LanguageCNN(nn.Module):
    def __init__(self, embedding_size, filter_size, hidden_size, sequence_state_size):
        super(LanguageCNN, self).__init__()
        self.conv = nn.Conv1d(in_channels=embedding_size,
                              out_channels=hidden_size,
                              kernel_size=filter_size,
                              padding=filter_size/2)
        self.linear = nn.Linear(hidden_size, sequence_state_size)

    def forward(self, x):
        x = F.tanh(self.conv(x))
        x = F.avg_pool1d(x, x.size()[-1])
        #return self.linear(x.squeeze())
        return x
