import torch.nn as nn
import torch.nn.functional as F
import torch


class GradientReversal(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -1e-5 * grad_output


class CNNDomainTransferNet(nn.Module):
    def __init__(self, feature_extractor):
        super(CNNDomainTransferNet, self).__init__()
        self.dropout = nn.Dropout(p=0.05)
        self.feature_extractor = feature_extractor
        self.linear = nn.Linear(667, 2)
        self.softmax = nn.Softmax()
        try:
            self.hidden_size = feature_extractor.hidden_size
        except Exception:
            pass

    def forward(self, x, return_domain=False):
        x = self.feature_extractor(x)
        if return_domain:
            x = GradientReversal.apply(x)
            return self.softmax(self.linear(x))
        return x


class LSTMDomainTransferNet(nn.Module):
    def __init__(self, feature_extractor):
        super(LSTMDomainTransferNet, self).__init__()
        self.dropout = nn.Dropout(p=0.05)
        self.feature_extractor = feature_extractor
        self.linear = nn.Linear(240, 2)
        self.softmax = nn.Softmax()
        try:
            self.hidden_size = feature_extractor.hidden_size
        except Exception:
            pass

    def forward(self, x, state, return_domain=False):
        x, state = self.feature_extractor(x, state)
        x = self.dropout(x)
        if return_domain:
            x = GradientReversal.apply(x)
            x = torch.transpose(x.squeeze(1), 0, 1).unsqueeze(0)
            x = F.avg_pool1d(x, x.size()[-1]).squeeze(2)
            return self.softmax(self.linear(x))
        return x, state
