import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, in_channels=3, n_classes=19):
        super().__init__()
        self.in_channels = in_channels

        # ---- Encoder ----
        self.down1 = Down(in_channels, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        self.down4 = Down(256, 512)

        # ---- Decoder ----
        self.up1 = Up(512 + 256, 256)
        self.up2 = Up(256 + 128, 128)
        self.up3 = Up(128 + 64, 64)
        self.up4 = Up(64 + 64, 64)

        self.outc = OutConv(64, n_classes)

    def forward(self, x):

        if x.shape[1] != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} channels but got {x.shape[1]}")

        # Encoder
        x, f2  = self.down1(x)
        x, f6  = self.down2(x)
        x, f10 = self.down3(x)
        x, f14 = self.down4(x)

        # Decoder
        x = self.up1(x, f14)
        x = self.up2(x, f10)
        x = self.up3(x, f6)
        x = self.up4(x, f2)

        return self.outc(x)


# -------------------------
# Residual DoubleConv (CHANGED)
# -------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

        self.relu = nn.ReLU(inplace=True)

        # match channels for residual
        self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        res = self.skip(x)
        x = self.conv(x)
        return self.relu(x + res)


# -------------------------
# Down (UNCHANGED structurally)
# -------------------------
class Down(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.2):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout2d(dropout)

    def forward(self, x):
        skip = self.conv(x)
        x = self.pool(skip)
        x = self.dropout(x)
        return x, skip


# -------------------------
# Attention Gate (NEW)
# -------------------------
class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()

        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, 1, bias=False),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, 1, bias=False),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, 1, bias=False),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)

        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi


# -------------------------
# Up (MINIMAL CHANGE: attention added)
# -------------------------
class Up(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.2):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dropout = nn.Dropout2d(dropout)

        # infer skip channels from concatenation
        self.att = AttentionGate(
            F_g=in_channels - out_channels,
            F_l=out_channels,
            F_int=out_channels // 2
        )

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        x = self.dropout(x)

        # Safe spatial alignment
        h = min(x.shape[2], skip.shape[2])
        w = min(x.shape[3], skip.shape[3])
        x = x[:, :, :h, :w]
        skip = skip[:, :, :h, :w]

        # APPLY ATTENTION (NEW)
        skip = self.att(x, skip)

        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)