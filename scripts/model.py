import torch
import torch.nn as nn
import torch.nn.functional as F

#Best accuracy - 61% (test) 83% train for 20 epochs
# Total params: >300k

class CIFAR10Model(nn.Module):

    def __init__(self, dropout_value=0.25):

        self.dropout_value = dropout_value  # dropout value

        super(CIFAR10Model, self).__init__()

        # Input Block
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32,
                      kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(self.dropout_value)
        )  # output_size = 32

        # CONVOLUTION BLOCK 1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64,
                      kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(self.dropout_value)
        )  # output_size = 32

        # TRANSITION BLOCK 1
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32,
                      kernel_size=(1, 1), padding=0, bias=False),
        )  # output_size = 32
        self.pool1 = nn.MaxPool2d(2, 2)  # output_size = 16

        # CONVOLUTION BLOCK 2
        # DEPTHWISE CONVOLUTION AND POINTWISE CONVOLUTION
        self.depthwise1 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64,
                      kernel_size=(3, 3), padding=0, groups=32, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(self.dropout_value)
        )  # output_size = 16
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128,
                      kernel_size=(1, 1), padding=0, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(self.dropout_value)
        )  # output_size = 16

        # TRANSITION BLOCK 2
        self.pool2 = nn.MaxPool2d(2, 2)  # output_size = 8


        # CONVOLUTION BLOCK 3
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128,
                      kernel_size=(3, 3), padding=4, dilation=2, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(self.dropout_value)
        )  # output_size = 11
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128,
                      kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(self.dropout_value)
        )  # output_size = 11

        # TRANSITION BLOCK 3
        self.pool3 = nn.MaxPool2d(2, 2)  # output_size = 5

        # OUTPUT BLOCK
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=5)
        )  # output_size = 1

        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128,
                      kernel_size=(1, 1), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Dropout(self.dropout_value)
        )

        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=10,
                      kernel_size=(1, 1), padding=0, bias=False),
        )

        self.dropout = nn.Dropout(self.dropout_value)

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.pool1(x)
        x = self.depthwise1(x)
        x = self.convblock4(x)
        x = self.pool2(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        x = self.pool3(x)
        x = self.gap(x)
        x = self.convblock7(x)
        x = self.convblock8(x)

        x = x.view(-1, 10)
        return F.log_softmax(x)

#Best accuracy - 83% (test) 88% train for 20 epochs
# with augmentaions - 79% test and 70% train
# Total params: 309,354
# Choose this as Baseline



class Net1(nn.Module):
    def __init__(self,dropout_value = 0.1):
        
        super(Net1,self).__init__()

        self.dropout_value = dropout_value

        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32,
                      kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(self.dropout_value)
        ) 
        #output - 32

        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64,
                      kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(self.dropout_value)
        )
        #output  - 32

        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32,
                      kernel_size=(3, 3), padding=1,stride = 2,  bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(self.dropout_value)
        ) #output = 16

        self.depthwise4 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32,
                      kernel_size=(3, 3), padding=1, groups=32, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(self.dropout_value)
        ) #output = 16

        self.pointwise4 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64,
                      kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(self.dropout_value)
        ) #output = 16

        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels = 64, out_channels=128,
                      kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(self.dropout_value)
        )
        #output = 16
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels = 128, out_channels=32,
                      kernel_size=(3, 3), padding=1,stride = 2, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(self.dropout_value)
        )
        #output = 8

        self.dilation7 = nn.Sequential(
            nn.Conv2d(in_channels = 32, out_channels=64,
                      kernel_size=(3, 3), padding=1,dilation=2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(self.dropout_value)
        )
        #output = 6

        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels = 64, out_channels=128,
                      kernel_size=(3, 3), bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(self.dropout_value)
        )
        #output = 4

        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=4)
            )# output size = 1 rf=34
        
        self.fc1 = nn.Linear(in_features=128,out_features=256)
        self.fc2 = nn.Linear(in_features=256,out_features=128)
        self.fc3 = nn.Linear(in_features=128,out_features=10)

    def forward(self,x):

        x = self.convblock1(x)
        x = self.convblock2(x)

        x = self.convblock3(x)
        
        x = self.depthwise4(x)
        x = self.pointwise4(x)
        x = self.convblock5(x)

        x = self.convblock6(x)
        
        x = self.dilation7(x)
        x = self.convblock8(x)

        x = self.gap(x)

        x = x.view(-1,128)
        
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        
        x = x.view(-1, 10)
        
        return F.log_softmax(x, dim=-1)


# Total params: 197,450
# reached 80% in the 33rd epoch and didnt cross 81 even after 61 epochs
class Net3(nn.Module):
    def __init__(self,dropout_value = 0.1):
        
        super(Net3,self).__init__()

        self.dropout_value = dropout_value

        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32,
                      kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(self.dropout_value)
        ) 
        #output - 32

        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64,  
                      kernel_size=(3, 3), padding=1,dilation = 2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(self.dropout_value)
        )
        #output  - 30

        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32,
                      kernel_size=(3, 3), padding=1,stride = 2,  bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(self.dropout_value)
        ) #output = 15

        self.depthwise4 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32,
                      kernel_size=(3, 3), padding=1, groups=32, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(self.dropout_value)
        ) #output = 15

        self.pointwise4 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64,
                      kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(self.dropout_value)
        ) #output = 15

        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels = 64, out_channels=64,
                      kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(self.dropout_value)
        )
        #output = 15
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels = 64, out_channels=32,
                      kernel_size=(3, 3), padding=1,stride = 2, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(self.dropout_value)
        )
        #output = 8

        self.dilation7 = nn.Sequential(
            nn.Conv2d(in_channels = 32, out_channels=64,
                      kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(self.dropout_value)
        )
        #output = 6

        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels = 64, out_channels=128,
                      kernel_size=(3, 3), bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(self.dropout_value)
        )
        #output = 4

        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=4)
            )# output size = 1 rf=34
        
        self.fc1 = nn.Linear(in_features=128,out_features=64)
        self.fc2 = nn.Linear(in_features=64,out_features=10)
        #self.fc3 = nn.Linear(in_features=128,out_features=10

    def forward(self,x):

        x = self.convblock1(x)
        x = self.convblock2(x)

        x = self.convblock3(x)
        
        x = self.depthwise4(x)
        x = self.pointwise4(x)
        x = self.convblock5(x)

        x = self.convblock6(x)
        
        x = self.dilation7(x)
        x = self.convblock8(x)

        x = self.gap(x)

        x = x.view(-1,128)
        
        x = self.fc1(x)
        x = self.fc2(x)
        #x = self.fc3(x)
        
        x = x.view(-1, 10)
        
        return F.log_softmax(x, dim=-1)


# Total params: 176,234


class Net4(nn.Module):
    def __init__(self,dropout_value = 0.02):
        
        super(Net4,self).__init__()

        self.dropout_value = dropout_value

        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32,
                      kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(self.dropout_value)
        ) 
        #output - 32 rf - 3

        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64,  
                      kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(self.dropout_value)
        )
        #output  - 32 rf - 5

        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32,
                      kernel_size=(3, 3), padding=1,stride = 2,  bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(self.dropout_value)
        ) #output = 16 rf - 7

        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64,
                      kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(self.dropout_value)
        ) #output = 16 rf = 11

        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels = 64, out_channels=96,
                      kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.Dropout(self.dropout_value)
        )
        #output = 16 rf = 15
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels = 96, out_channels=32,
                      kernel_size=(3, 3), padding=1,stride = 2, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(self.dropout_value)
        )
        #output = 8 rf = 19

        self.dilation7 = nn.Sequential(
            nn.Conv2d(in_channels = 32, out_channels=64,
                      kernel_size=(3, 3), padding=1,dilation = 2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(self.dropout_value)
        )
        #output = 6 rf = 27

        self.depthwise8 = nn.Sequential(
            nn.Conv2d(in_channels = 64, out_channels = 64,
                      kernel_size=(3, 3),groups=64, bias=False),
            nn.Conv2d(in_channels = 64, out_channels = 128,
                      kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        #output = 4

        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=4)
            )# output size = 1 
        
        self.fc1 = nn.Linear(in_features=128,out_features=64)
        self.fc2 = nn.Linear(in_features=64,out_features=10)
        #self.fc3 = nn.Linear(in_features=128,out_features=10

    def forward(self,x):

        x = self.convblock1(x)
        x = self.convblock2(x)

        x = self.convblock3(x)
        
        x = self.convblock4(x)
        x = self.convblock5(x)

        x = self.convblock6(x)
        
        x = self.dilation7(x)
        x = self.depthwise8(x)

        x = self.gap(x)

        x = x.view(-1,128)
        
        x = self.fc1(x)
        x = self.fc2(x)
        #x = self.fc3(x)
        
        x = x.view(-1, 10)
        
        return F.log_softmax(x, dim=-1)


