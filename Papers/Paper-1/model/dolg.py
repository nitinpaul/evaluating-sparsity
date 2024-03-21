import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import DataLoader
from pytorch_lightning import LightningModule

from config import Config
from model.gem_pool import GeM
from model.arcface import ArcFace
from dataset.dataset import LmkRetrDataset


class MultiAtrous(nn.Module):
    """Multi-Atrous Convolution Module."""

    def __init__(self, in_channel, out_channel, size, dilation_rates=[3, 6, 9]):
        """
        Args:
            in_channel: Number of input channels.
            out_channel: Number of output channels.
            size: Output size (height and width).
            dilation_rates: List of dilation rates for the convolutions.
        """
        super().__init__()

        self.dilated_convs = [
            nn.Conv2d(
                in_channel, int(out_channel / 4), kernel_size=3, dilation=rate, padding=rate
            )
            for rate in dilation_rates
        ]

        # GAP branch
        self.gap_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channel, int(out_channel / 4), kernel_size=1),
            nn.ReLU(),
            nn.Upsample(size=(size, size), mode="bilinear"),
        )
        self.dilated_convs.append(self.gap_branch)
        self.dilated_convs = nn.ModuleList(self.dilated_convs) 

    def forward(self, x):
        """Computes the forward pass of the Multi-Atrous module.""" 
        local_feat = []
        for dilated_conv in self.dilated_convs:
            local_feat.append(dilated_conv(x))
        local_feat = torch.cat(local_feat, dim=1)
        return local_feat

class DolgLocalBranch(nn.Module):
    """DOLG Local Branch Module."""

    def __init__(self, in_channel, out_channel, hidden_channel=2048):
        """
        Args:
            in_channel: Number of input channels.
            out_channel: Number of output channels.
            hidden_channel: Number of channels in intermediate layers.
        """
        super().__init__()
        self.multi_atrous = MultiAtrous(
            in_channel, hidden_channel, size=int(Config.image_size / 8)
        )
        self.conv1x1_1 = nn.Conv2d(hidden_channel, out_channel, kernel_size=1)
        self.conv1x1_2 = nn.Conv2d(out_channel, out_channel, kernel_size=1, bias=False)
        self.conv1x1_3 = nn.Conv2d(out_channel, out_channel, kernel_size=1)

        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(out_channel)
        self.softplus = nn.Softplus()

    def forward(self, x):
        """Computes the forward pass of the DOLG Local Branch module."""
        local_feat = self.multi_atrous(x)
        local_feat = self.conv1x1_1(local_feat)
        local_feat = self.relu(local_feat)
        local_feat = self.conv1x1_2(local_feat)
        local_feat = self.bn(local_feat)

        attention_map = self.relu(local_feat)
        attention_map = self.conv1x1_3(attention_map)
        attention_map = self.softplus(attention_map)

        local_feat = F.normalize(local_feat, p=2, dim=1)
        local_feat = local_feat * attention_map
        return local_feat

class DolgNet(LightningModule):
    """DOLG Network Module (Pytorch Lightning)."""

    def __init__(self, input_dim, hidden_dim, output_dim, num_of_classes):
        """
        Args:
            input_dim: Input dimension (number of channels).
            hidden_dim: Dimension for hidden layers.
            output_dim: Output dimension (number of embedding features).
            num_of_classes: Number of classes for classification.
        """
        super().__init__()
        self.cnn = timm.create_model(
            "tv_resnet101",
            pretrained=True,
            features_only=True,
            in_chans=input_dim,
            out_indices=(2, 3),
        )
        self.orthogonal_fusion = OrthogonalFusion()
        self.local_branch = DolgLocalBranch(512, hidden_dim)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gem_pool = GeM()
        self.fc_1 = nn.Linear(1024, hidden_dim)
        self.fc_2 = nn.Linear(int(2 * hidden_dim), output_dim)

        self.criterion = ArcFace(
            in_features=output_dim,
            out_features=num_of_classes,
            scale_factor=30,
            margin=0.15,
            criterion=nn.CrossEntropyLoss(),
        )
        self.lr = Config.lr

    def forward(self, x):
        """Computes the forward pass of the DOLG network."""
        output = self.cnn(x)

        local_feat = self.local_branch(output[0])  # ,hidden_channel,16,16
        global_feat = self.fc_1(self.gem_pool(output[1]).squeeze())  # ,1024

        feat = self.orthogonal_fusion(local_feat, global_feat)
        feat = self.gap(feat).squeeze()
        feat = self.fc_2(feat)
        return feat

    def training_step(self, batch, batch_idx):
        img, label, _ = batch
        embd = self(img)
        loss, logits = self.criterion(embd, label)
        return loss

    def configure_optimizers(self):
        optimizer = optim.SGD(
            self.parameters(), lr=self.lr, momentum=0.9, weight_decay=1e-5
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000)
        return [optimizer], [scheduler]

    def train_dataloader(self):
        dataset = LmkRetrDataset()
        return DataLoader(
            dataset,
            batch_size=Config.train_batch_size,
            num_workers=Config.num_workers,
            shuffle=True,
            pin_memory=True,
            persistent_workers=True,
        )
