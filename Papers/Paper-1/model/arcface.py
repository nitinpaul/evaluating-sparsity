import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcFace(nn.Module):
    """ArcFace (Additive Angular Margin Loss) module."""

    def __init__(self, in_features, out_features, scale_factor=64.0, margin=0.50, criterion=None):
        """
        Args:
            in_features: Size of input feature vectors.
            out_features: Number of output classes.
            scale_factor: Scaling factor for the logits.
            margin: Angular margin for ArcFace.
            criterion: Loss function (default: nn.CrossEntropyLoss).
        """
        super(ArcFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        if criterion:
            self.criterion = criterion
        else:
            self.criterion = nn.CrossEntropyLoss()

        self.margin = margin
        self.scale_factor = scale_factor

        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(self, input, label):
        # Input is not L2 normalized
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))

        phi = cosine * self.cos_m - sine * self.sin_m  
        phi = phi.type(cosine.type())  # Ensure consistent type
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros(cosine.size(), device=input.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)

        logit = (one_hot * phi) + ((1.0 - one_hot) * cosine)  
        logit *= self.scale_factor

        loss = self.criterion(logit, label)
        return loss, logit
