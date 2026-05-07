import torch
from torch import nn
from torch.optim import SGD,Adam

from src.dataset import get_cifar10_dataloaders
from src.utils import train_loop, test_loop, init_weights
from models.resnet import ResNet
from models.plainnet import PlainNet

def train_and_test(model,loss_fn=nn.CrossEntropyLoss(),train_dataloader,test_dataloader,train_acc_list=[],train_loss_list=[],test_acc_list=[],
                   device='cpu',optimizers=[Adam,Adam,SGD],epochs=[20,15,15],lrs=[1e-3,1e-4,1e-4],weight_decays=[1e-3,1e-3,1e-3])
  epoch=0
  for i in range(len(epochs)):
    now_epoch=epochs[i]
    lr = lrs[i]
    weight_decay = weight_decays[i]
    optimizer = optimizers[i](model.parameters(),lr=lr,weight_decay = weight_decay)

    for _ in range(now_epoch):
      print(f"-----epoch {epoch+1}-----")
      train_loop(model,
                optimizer,
                loss_fn,
                train_dataloader,
                train_acc_list,
                train_loss_list,
                device
                )
      test_loop(model,
                test_dataloader,
                test_acc_list,
                device
                )
    epoch+=1

device = 'cuda' if torch.cuda.is_avaliable() else 'cpu'

# =========================
# Get dataloader
# =========================
train_dataloader, test_dataloader = get_cifar10_dataloaders(batch_size=128, root='./data')

# =========================
# Model select
# =========================
ResNet44=ResNet(N=7).to(device)
ResNet44.apply(init_weights)
#PlainNet44=PlainNet(N=7).to(device)
#PlainNet44.apply(init_weights)

model = ResNet44
#model = PlainNet44

# =========================
# Accuracy and loss list
# =========================
Res_train_acc_list = []
Res_test_acc_list = []
Res_train_loss_list = []

#Plain_train_acc_list = []
#Plain_test_acc_list = []
#Plain_train_loss_list = []

# =========================
# Training conditions
# =========================
loss_fn = nn.CrossEntropyLoss()
epochs=[20,15,15]
lrs=[1e-3,1e-4,1e-4]
weight_decays =[1e-3,1e-3,1e-3]
optimizers=[Adam,Adam,SGD]

# =========================
# Train and test
# =========================
train_and_test(model,loss_fn=nn.CrossEntropyLoss(),train_dataloader,test_dataloader,train_acc_list=[],train_loss_list=[],test_acc_list=[],
                   device='cuda',optimizers=[Adam,Adam,SGD],epochs=[20,35,50],lrs=[1e-3,1e-4,1e-4],weight_decays=[1e-3,1e-3,1e-3])
