import torch.nn as nn
import torch.nn.functional as F


class LanguageCNN(nn.Module):
    def __init__(self, embedding_size, sequence_state_size):
        super(LanguageCNN, self).__init__()
        self.conv = nn.Conv1d(in_channels=embedding_size,
                              out_channels=1,
                              kernel_size=filter_size)
        self.sequence_state_size = sequence_state_size

    def forward(self, x):
        x = F.tanh(self.conv(x))
        stride_and_kernel_size = len(x)/sequence_state_size
        return F.avg_pool1d(x, stride_and_kernel_size, stride_and_kernel_size)


model = LanguageCNN(200, 5)
print model.state_dict()
from torch.autograd import Variable
import torch
print model(Variable(torch.randn(1, 200, 6)))
print model.state_dict()
