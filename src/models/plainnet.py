import torch
from torch import nn

class PlainNet(nn.Module):
  """
  Predict 3*32*32 CIFAR10 dataset
  Same architecture to ResNet except skip connection
  Conv * (2N+1) (3*3 kernel, filter size = 16) -
  Conv * 2N (3*3 kernel, filter size = 32) -
  Conv * 2N (3*3 kernel, filter size = 64) -
  Average pooling - FC
  """
  def __init__(self,N):
    super().__init__()
    self.N = N
    filter_size = 16

    self.Pre = nn.Sequential(
      nn.Conv2d(3,16,(3,3),padding=1,bias=False),
      nn.BatchNorm2d(filter_size),
      nn.ReLU()
    )

    module_dict={}
    for i in range(1,4):
      for n in range(1,self.N*2+1):
        if i!=1 and n==1: # halve feature map size
          #module_dict[f"Change_Dim{i}"] = nn.Sequential(
          #  nn.Conv2d(filter_size,filter_size*2,(1,1),stride=2,bias=False),
          #  nn.BatchNorm2d(filter_size*2)
          #)
          module_dict[f"Conv{i}_{n}"] = nn.Conv2d(filter_size,filter_size*2,(3,3),stride=2,padding=1,bias=False)
          filter_size*=2
        else:
          module_dict[f"Conv{i}_{n}"] = nn.Conv2d(filter_size,filter_size,(3,3),stride=1,padding=1,bias=False)
        module_dict[f"Batchnorm{i}_{n}"] = nn.BatchNorm2d(filter_size)
        module_dict[f"ReLU{i}_{n}"] = nn.ReLU()

    module_dict["Avgpool"] = nn.AdaptiveAvgPool2d((1,1))
    module_dict["Flatten"] = nn.Flatten()
    module_dict["FC"] = nn.Linear(64,10)

    self.ModuleDict = nn.ModuleDict(module_dict)

  def forward(self,x):
    x = self.Pre(x)
    for i in range(1,4):
      for n in range(1,self.N*2+1,2):
        y = self.ModuleDict[f"Conv{i}_{n}"](x)
        y = self.ModuleDict[f"Batchnorm{i}_{n}"](y)
        y = self.ModuleDict[f"ReLU{i}_{n}"](y)

        y = self.ModuleDict[f"Conv{i}_{n+1}"](y)
        y = self.ModuleDict[f"Batchnorm{i}_{n+1}"](y)
        #if i!=1 and n==1:
          #x = self.ModuleDict[f"Change_Dim{i}"](x)
        #x = self.ModuleDict[f"ReLU{i}_{n+1}"](y+x)
        x = self.ModuleDict[f"ReLU{i}_{n+1}"](y)

    x = self.ModuleDict["Avgpool"](x)
    x = self.ModuleDict["Flatten"](x)
    x = self.ModuleDict["FC"](x)

    return x
