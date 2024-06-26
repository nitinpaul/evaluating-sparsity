import torch
import torch.nn as nn
import torch.nn.functional as F


class GeM(nn.Module):
    """Generalized Mean Pooling (GeM) module."""

    def __init__(self, p=3, eps=1e-6, requires_grad=False):
        """
        Args:
            p: Power parameter for GeM pooling (default: 3).
            eps: Small value for numerical stability (default: 1e-6).
            requires_grad: Whether the p parameter requires gradients.
        """
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p, requires_grad=requires_grad)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        return F.avg_pool2d(
            x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))
        ).pow(1.0 / p)

    def __repr__(self):
        return (
            self.__class__.__name__
            + "("
            + "p="
            + "{:.4f}".format(self.p.data.tolist()[0])
            + ", "
            + "eps="
            + str(self.eps)
            + ")"
        )

