import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv = nn.Conv1d(200, 667, 3, 1, 2)

    def forward(self, x):
        x = F.tanh(self.conv(x))
        x = F.avg_pool1d(x, x.size()[-1])
        return x.squeeze(2)
