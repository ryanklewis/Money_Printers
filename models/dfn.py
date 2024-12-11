import torch
import torch.nn as nn


class DFN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, 512)
        self.layer2 = nn.Linear(512, 256)
        self.layer3 = nn.Linear(256, 512)
        self.layer4 = nn.Linear(512, 128)
        self.layer5 = nn.Linear(128, 64)
        self.layer6 = nn.Linear(64, 3)

        self.dropout = nn.Dropout(0.03)
        self.relu = nn.ReLU()
        self.norm1 = nn.InstanceNorm1d(512)
        self.norm2 = nn.InstanceNorm1d(256)
        self.norm3 = nn.InstanceNorm1d(512)

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)

    def forward(self, x):
        inter1 = self.norm1(self.leaky_relu(self.layer1(x)))
        inter2 = self.dropout(inter1)
        inter3 = self.norm2(self.leaky_relu(self.layer2(inter2)))
        inter4 = self.dropout(inter3)
        inter5 = self.norm3(self.leaky_relu(self.layer3(inter4)))
        inter6 = self.dropout(inter5)
        inter7 = self.leaky_relu(self.layer4(inter6))
        inter8 = self.dropout(inter7)
        inter9 = self.leaky_relu(self.layer5(inter8))
        out = self.layer6(inter9)
        return out
