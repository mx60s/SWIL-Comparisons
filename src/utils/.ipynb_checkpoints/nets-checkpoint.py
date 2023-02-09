import torch
from torch import nn
import torch.nn.functional as F


class LinearFashionMNIST(nn.Module):
  def __init__(self, num_classes : int):
    super(LinearFashionMNIST, self).__init__()

    self.flatten = nn.Flatten()
  
    self.linear_stack = nn.Sequential(
        nn.Linear(28*28, 128),
        nn.Linear(128, num_classes)
    )

  def forward(self, x):
    x = self.flatten(x)
    logits = self.linear_stack(x)
    return logits

  
class LinearFashionMNIST_alt(nn.Module):
  def __init__(self, input_size, num_classes: int):
    super(LinearFashionMNIST_alt, self).__init__()

    self.flatten = nn.Flatten()
    self.input_layer = nn.Linear(input_size, 128)
    self.output_layer = nn.Linear(128, num_classes)

  def forward(self, x):
    x = self.flatten(x)
    #x = torch.flatten(x)
    return self.output_layer(self.input_layer(x))


# TODO: add L2 regularization for CIFAR10
# changing kernel size to 2 -- why was it set to 5? 
class CIFAR10Cnn(nn.Module):
    def __init__(self, num_classes : int):
        self.num_classes = num_classes
        super().__init__()
        self.conv1 = nn.Conv2d(3,6,5)
        self.norm1 = nn.BatchNorm2d(6)
        self.conv2 = nn.Conv2d(6,16,5)
        self.norm2 = nn.BatchNorm2d(16)
        
        self.conv3 = nn.Conv2d(16,32,5)
        self.norm3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32,64,5)
        self.norm4 = nn.BatchNorm2d(64)
        
        self.conv5 = nn.Conv2d(64,128,5)
        self.norm5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128,256,5)
        self.norm6 = nn.BatchNorm2d(256)
        
        self.pool = nn.MaxPool2d((5,5))
        #self.pool2 = nn.MaxPool2d((5,5))
        
        self.fc1 = nn.Linear(1024, self.num_classes)
        
    def forward(self, x):
        out = F.relu(self.norm1(self.conv1(x)))
        print(out.shape)
        out = F.relu(self.norm2(self.conv2(out)))
        print(out.shape)
        out = self.pool(out)
        print(out.shape)
        
        out = F.relu(self.norm3(self.conv3(out)))
        out = F.relu(self.norm4(self.conv4(out)))
        out = self.pool(out)
        print(out.shape)
        
        out = F.relu(self.norm5(self.conv5(out)))
        out = F.relu(self.norm6(self.conv6(out)))
        out = self.pool(out)

        out = torch.flatten(out, 1)
        
        out = self.fc1(out)
        return out
    

# TODO: add L2 regularization for CIFAR10
# changing kernel size to 2 -- why was it set to 5?
class CNN_6L(nn.Module):
    def __init__(self, num_classes: int, input_channels=3):
        super(CNN_6L, self).__init__()
        self.num_classes = num_classes
        
        '''Convolutional Block 1'''
        self.conv_block1 = nn.Sequential()
        # First Convolution
        self.conv_block1.add_module("Conv1", nn.Conv2d(
            in_channels=input_channels, out_channels=16, kernel_size=3, padding=1))
        self.conv_block1.add_module("Relu1", nn.ReLU())
        self.conv_block1.add_module("BN1", nn.BatchNorm2d(num_features=16))
        # Second Convolution
        self.conv_block1.add_module("Conv2", nn.Conv2d(
            in_channels=16, out_channels=16, kernel_size=3, padding=1))
        self.conv_block1.add_module("Relu2", nn.ReLU())
        self.conv_block1.add_module("BN2", nn.BatchNorm2d(num_features=16))
        self.conv_block1.add_module("Pool1", nn.MaxPool2d((2, 2)))

        '''Convolutional Block 2'''
        self.conv_block2 = nn.Sequential()
        # Third Convolution
        self.conv_block2.add_module("Conv3", nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=3, padding=1))
        self.conv_block2.add_module("Relu3", nn.ReLU())
        self.conv_block2.add_module("BN3", nn.BatchNorm2d(num_features=32))
        # Fourth Convolution
        self.conv_block2.add_module("Conv4", nn.Conv2d(
            in_channels=32, out_channels=32, kernel_size=3, padding=1))
        self.conv_block2.add_module("Relu4", nn.ReLU())
        self.conv_block2.add_module("BN4", nn.BatchNorm2d(num_features=32))
        self.conv_block2.add_module("Pool2", nn.MaxPool2d((2, 2)))

        '''Convolutional Block 3'''
        self.conv_block3 = nn.Sequential()
        # Fifth Convolution
        self.conv_block3.add_module("Conv5", nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, padding=1))
        self.conv_block3.add_module("Relu5", nn.ReLU())
        self.conv_block3.add_module("BN5", nn.BatchNorm2d(num_features=64))
        # Sixth Convolution
        self.conv_block3.add_module("Conv6", nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, padding=1))
        self.conv_block3.add_module("Relu6", nn.ReLU())
        self.conv_block3.add_module("BN6", nn.BatchNorm2d(num_features=64))
        self.conv_block3.add_module("Pool3", nn.MaxPool2d((2, 2)))
        
        # Output Layer
        self.fc1 = nn.Linear(64 * 4 * 4, self.num_classes)

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)

        return x


class CNN_demo(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
