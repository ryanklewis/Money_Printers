import torch.nn as nn
import torch


class TCN(nn.Module):
    def __init__(self, input_channels, num_classes=3):
        super(TCN, self).__init__()

        self.conv1 = nn.Conv1d(
            input_channels, 64, kernel_size=3, dilation=1, padding=1)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=3, dilation=2, padding=2)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, dilation=4, padding=4)
        self.conv4 = nn.Conv1d(128, 128, kernel_size=3, dilation=8, padding=8)

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        inter1 = torch.relu(self.conv1(x))
        inter2 = torch.relu(self.conv2(inter1))
        inter3 = torch.relu(self.conv3(inter2))
        inter4 = torch.relu(self.conv4(inter3))

        inter5 = self.pool(inter4)

        inter6 = inter5.view(inter5.size(0), -1)

        inter7 = torch.relu(self.fc1(inter6))
        out = self.fc2(inter7)

        return out
