import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):

    def __init__(self, channels, reduction=8):

        super().__init__()

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):

        b, c, _ = x.shape

        y = self.pool(x).view(b, c)

        y = self.fc(y).view(b, c, 1)

        return x * y


class ResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels):

        super().__init__()

        self.conv1 = nn.Conv1d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)

        self.conv2 = nn.Conv1d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.se = SEBlock(out_channels)

        if in_channels != out_channels:
            self.shortcut = nn.Conv1d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):

        residual = self.shortcut(x)

        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))

        x = self.se(x)

        x = x + residual

        return F.relu(x)


class UNet1D(nn.Module):

    def __init__(self, in_channels=5):

        super().__init__()

        self.encoder1 = ResidualBlock(in_channels, 32)
        self.pool1 = nn.MaxPool1d(2)

        self.encoder2 = ResidualBlock(32, 64)
        self.pool2 = nn.MaxPool1d(2)

        self.encoder3 = ResidualBlock(64, 128)
        self.pool3 = nn.MaxPool1d(2)

        self.bottleneck = ResidualBlock(128, 256)

        self.up3 = nn.ConvTranspose1d(256, 128, 2, stride=2)
        self.decoder3 = ResidualBlock(256, 128)

        self.up2 = nn.ConvTranspose1d(128, 64, 2, stride=2)
        self.decoder2 = ResidualBlock(128, 64)

        self.up1 = nn.ConvTranspose1d(64, 32, 2, stride=2)
        self.decoder1 = ResidualBlock(64, 32)

        self.final = nn.Conv1d(32, 1, kernel_size=1)

    def forward(self, x):

        e1 = self.encoder1(x)
        p1 = self.pool1(e1)

        e2 = self.encoder2(p1)
        p2 = self.pool2(e2)

        e3 = self.encoder3(p2)
        p3 = self.pool3(e3)

        b = self.bottleneck(p3)

        d3 = self.up3(b)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.decoder3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.decoder2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.decoder1(d1)

        out = self.final(d1)

        return torch.sigmoid(out)