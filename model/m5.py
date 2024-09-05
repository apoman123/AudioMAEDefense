import torch
import torch.nn as nn

class Convlayer(nn.Module):
    def __init__(self, in_channels, out_chanels, kernel, stride):
        super(Convlayer, self).__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_chanels,
            kernel_size=kernel,
            stride=stride
            )
        self.batch_norm = nn.BatchNorm1d(out_chanels)
        self.relu = nn.ReLU()
    def forward(self, input_data):
        result = self.conv(input_data)
        result = self.batch_norm(result)
        return self.relu(result)
    

class M5Net(nn.Module):
    def __init__(self, num_classes):
        super(M5Net, self).__init__()
        self.input_conv = Convlayer(1, 128, 80, 4)
        self.max_pooling = nn.MaxPool1d(4, 1)

        self.conv1 = Convlayer(128, 128, 3, 1)
        self.conv2 = Convlayer(128, 256, 3, 1)
        self.conv3 = Convlayer(256, 512, 3, 1)

        self.global_avg_pooling = nn.AdaptiveAvgPool1d(1)
        
        self.softmax = nn.Softmax(dim=-1)
        self.linear = nn.Linear(512, num_classes)
        
    def forward(self, input_data):
        result = self.max_pooling(self.input_conv(input_data))
        result = self.max_pooling(self.conv1(result))
        result = self.max_pooling(self.conv2(result))
        result = self.max_pooling(self.conv3(result))
        result = self.global_avg_pooling(result).squeeze(-1)
        result = self.linear(result)
        return self.softmax(result)