import torch
import torch.nn as nn
import torch.nn.functional as F

dropout_value = 0.02

def depthwise_separable_conv(nin, ks, p):
  return nn.Sequential(
      nn.Conv2d(nin, nin, kernel_size=ks, padding=p, groups=nin, bias=False), 
      nn.BatchNorm2d(nin),
      nn.ReLU())

def dilated_conv(nin, nout, ks, p):
  return nn.Sequential(
      nn.Conv2d(nin, nout, kernel_size=ks, padding=p, dilation=2, bias=False), 
      nn.BatchNorm2d(nout),
      nn.ReLU())
  
class Net(nn.Module):
    def init(self):

        super(Net,self).init()

        # Convolution Block 1
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=32,kernel_size=(3,3),padding = 1, bias = False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_value), 

            depthwise_separable_conv(32, 3, 0),
            depthwise_separable_conv(32, 3, 0), 
            depthwise_separable_conv(32, 3, 1)          
        )
        
        # Convolution Block 2
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=32,out_channels=32,kernel_size=(3,3),padding = 1, bias = False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_value), 

            dilated_conv(32, 32, 3, 0), 
            dilated_conv(32, 32, 3, 0),
            dilated_conv(32, 32, 3, 0),
        )
        
        # Convolution Block 3
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=32,out_channels=32,kernel_size=(3,3),padding = 1, bias = False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_value), 

            dilated_conv(32, 32, 3, 1), 
            dilated_conv(32, 32, 3, 1),
            dilated_conv(32, 32, 3, 1),

            depthwise_separable_conv(32, 3, 1),       
            depthwise_separable_conv(32, 3, 1),       
            depthwise_separable_conv(32, 3, 1),       
        )

        # Convolution Block 4
        self.convblock4 = nn.Sequential(
            
            depthwise_separable_conv(32, 3, 1),       
            depthwise_separable_conv(32, 3, 1),       
            depthwise_separable_conv(32, 3, 1),       
          
            nn.Conv2d(in_channels=32,out_channels=16,kernel_size=(3,3),padding = 0, bias = False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(dropout_value), 

            nn.Conv2d(in_channels=16,out_channels=16,kernel_size=(3,3),padding = 1, bias = False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(dropout_value), 
        )

        # Global average pooling layer
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=8),
            nn.Conv2d(in_channels=16,out_channels=10,kernel_size=(1,1),padding = 0, bias = False),
        )

    def forward(self, x):

        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.convblock4(x)

        x = self.gap(x)
        x = x.view(-1,10)

        return F.log_softmax(x,dim = -1)