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


class CNN_3B(nn.Module):
    def __init__(self, num_classes: int, input_channels=3):
        super(CNN_3B, self).__init__()
        self.num_classes = num_classes
        self.latent_training = False
        self.latent_eval = False
        self.flatten = nn.Flatten()
        
        # Convolutional Block 1
        self.conv_block1 = nn.Sequential()
        # First Convolution
        self.conv_block1.add_module("conv1", nn.Conv2d(
            in_channels=input_channels, out_channels=32, kernel_size=3, padding=1))
        self.conv_block1.add_module("relu1", nn.ReLU())
        self.conv_block1.add_module("bn1", nn.BatchNorm2d(num_features=32))
        # Second Convolution
        self.conv_block1.add_module("conv2", nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, padding=1))
        self.conv_block1.add_module("relu2", nn.ReLU())
        self.conv_block1.add_module("bn2", nn.BatchNorm2d(num_features=64))
        self.conv_block1.add_module("mpool", nn.MaxPool2d((2, 2)))
                                    
        # Convolutional Block 2
        self.conv_block2 = nn.Sequential()
        # Third Convolution
        self.conv_block2.add_module("conv1", nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=3, padding=1))
        self.conv_block2.add_module("relu1", nn.ReLU())
        self.conv_block2.add_module("bn1", nn.BatchNorm2d(num_features=128))
        # Fourth Convolution
        self.conv_block2.add_module("conv2", nn.Conv2d(
            in_channels=128, out_channels=128, kernel_size=3, padding=1))
        self.conv_block2.add_module("relu2", nn.ReLU())
        self.conv_block2.add_module("bn2", nn.BatchNorm2d(num_features=128))
        self.conv_block2.add_module("mpool", nn.MaxPool2d((2, 2)))
        
        # Convolutional Block 3
        self.conv_block3 = nn.Sequential()
        # Fifth Convolution
        self.conv_block3.add_module("conv1", nn.Conv2d(
            in_channels=128, out_channels=256, kernel_size=3, padding=1))
        self.conv_block3.add_module("relu1", nn.ReLU())
        self.conv_block3.add_module("bn1", nn.BatchNorm2d(num_features=256))
        # Sixth Convolution
        self.conv_block3.add_module("conv2", nn.Conv2d(
            in_channels=256, out_channels=256, kernel_size=3, padding=1))
        self.conv_block3.add_module("relu2", nn.ReLU())
        self.conv_block3.add_module("bn2", nn.BatchNorm2d(num_features=256))
        self.conv_block3.add_module("mpool", nn.MaxPool2d((2, 2)))
        
        # Fully Connected Block
        self.fc_block = nn.Sequential()
        self.fc_block.add_module("drop1", nn.Dropout(p=0.1))
        
        # First Linear
        self.fc_block.add_module("fc1", nn.Linear(256 * 4 * 4, 1024))
        self.fc_block.add_module("relu1", nn.ReLU())
        
        # Second Linear
        self.fc_block.add_module("fc2", nn.Linear(1024, 512))
        self.fc_block.add_module("relu2", nn.ReLU())
        self.fc_block.add_module("drop2", nn.Dropout(p=0.1))
        
        # Output Layer
        self.fc_block.add_module("fc3", nn.Linear(512, self.num_classes))
        
    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.flatten(x)
        x = self.fc_block(x)
        return x
    
    def latent_train(self):
        self.train()