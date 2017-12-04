import torch.nn as nn
import torch


class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg()


class DomainTransferNet(nn.Module):
    def __init__(self, feature_extractor, label_predictor, domain_classifier):
        super(DomainTransferNet, self).__init__()
        self.feature_extractor = feature_extractor
        self.label_predictor = label_predictor
        self.domain_classifier = domain_classifier

    def forward(self, x):
        x = self.feature_extractor(x)
        domain = GradientReversal.apply(x)
        domain = self.domain_classifier(domain)
        label = self.label_predictor(x)
        return label, domain
