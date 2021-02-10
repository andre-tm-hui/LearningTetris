import torch.nn as nn
import torch
from data import BOARD_DQN, FEATURE_DQN

class DeepQNetwork(nn.Module):
    def __init__(self, input_type = BOARD_DQN, input_size = (10,21)):
        super(DeepQNetwork, self).__init__()
        self.input_type = input_type

        if self.input_type == BOARD_DQN:
            self.conv1 = nn.Sequential(nn.Conv2d(1, 8, kernel_size=5, stride=1), nn.BatchNorm2d(8), nn.ReLU(inplace=True))
            self.conv2 = nn.Sequential(nn.Conv2d(8, 16, kernel_size=5, stride=1), nn.BatchNorm2d(16), nn.ReLU(inplace=True))
            
            def conv2d_size_out(size, kernel_size = 5, stride = 1):
            	return (size - (kernel_size - 1) - 1) // stride + 1
            convw = conv2d_size_out(conv2d_size_out(10))
            convh = conv2d_size_out(conv2d_size_out(21))
            linear_input_size = convw * convh * 16

            self.conv3 = nn.Sequential(nn.Linear(linear_input_size, 64), nn.ReLU(inplace=True))
        else:
            self.conv3 = nn.Sequential(nn.Linear(input_size[0], 64), nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(nn.Linear(64, 64), nn.ReLU(inplace=True))
        self.conv5 = nn.Sequential(nn.Linear(64, 1))

        self._create_weights()

    def _create_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        if self.input_type == BOARD_DQN:
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x.view(x.size(0),-1))
        else:
            x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x

class DeepQNetwork_features(nn.Module):
    def __init__(self):
        super(DeepQNetwork_features, self).__init__()

        self.conv1 = nn.Sequential(nn.Linear(8, 64), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Linear(64, 64), nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(nn.Linear(64, 1))

        self._create_weights()

    def _create_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x