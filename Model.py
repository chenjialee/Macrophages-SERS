import torch
import torch.nn as nn


class SimplifiedCNN(nn.Module):
    def __init__(self):
        super(SimplifiedCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, stride=2),  # 减少卷积层数量
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),  # 加入池化层减少尺寸


            nn.Conv1d(16, 32, kernel_size=5, stride=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(32, 64, kernel_size=5, stride=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(64, 128, kernel_size=3, stride=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )

        # 全连接层
        self.fc1 = nn.Linear(3456, 512)  # 根据flatten后的输出调整输入大小
        self.fc_out = nn.Linear(512, 3)  # 输出层
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        if x.dim() == 4:
            x = x.squeeze(1)  # 从 [8,1,1,1015] -> [8,1,1015]

        x = self.conv_layers(x)
        x = torch.flatten(x, 1)  # Flatten输出
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.dropout(x)
        x = self.fc_out(x)
        # x = self.dropout(x)
        return x