import torch.nn as nn
import torch.nn.functional as F


class ValveCNN(nn.Module):
    def __init__(self, n_classes=4):
        super().__init__()
        self.conv1 = nn.Conv1d(2, 16, kernel_size=15, stride=3)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=9, stride=3)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=5, stride=3)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(64, n_classes)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool(x).squeeze(-1)
        x = self.fc(x)
        return x