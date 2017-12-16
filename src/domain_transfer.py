import torch.nn as nn
import torch.nn.functional as F
import torch


class GradientReversal(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -1e-5*grad_output


class DomainTransferNet(nn.Module):
    def __init__(self, feature_extractor):
        super(DomainTransferNet, self).__init__()
        # self.dropout = nn.Dropout(p=0.05)
        self.feature_extractor = feature_extractor
        self.linear = nn.Linear(667, 2)
        self.softmax = nn.Softmax()

    def forward(self, x, return_domain=False):
        x = self.feature_extractor(x)
        if return_domain:
            x = GradientReversal.apply(x)
            return self.softmax(self.linear(x))
        return x
