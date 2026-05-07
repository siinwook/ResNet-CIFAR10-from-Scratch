import torch
from torchvision.datasets import CIFAR10
from torchvision.transforms import v2
from torch.utils.data import Dataset,DataLoader

def get_cifar10_dataloaders(batch_size=128,root='./data'):
  #CIFAR10 download mirror server
  #CIFAR10.url = "https://data.brainchip.com/dataset-mirror/cifar10/cifar-10-python.tar.gz"

  # =========================
  # Dataset transform
  # =========================
  train_transform = v2.Compose([
      # Train data augmentation
      v2.RandomCrop(32,padding=4),
      v2.RandomHorizontalFlip(),
      v2.ColorJitter(0.4,0.4,0.4,0.1), # Brightness, ...
      v2.RandomGrayscale(0.2), # RGB -> Gray
      v2.ToTensor()
  ])

  test_transform = v2.Compose([
      v2.ToTensor()
  ])

  # =========================
  # Dataset
  # =========================
  train_dataset = CIFAR10(
      root = root,
      train = True,
      transform= train_transform,
      download = True
  )

  test_dataset = CIFAR10(
      root = root,
      train = False,
      transform= test_transform,
      download = True
  )

  # =========================
  # Dataloader
  # =========================
  train_dataloader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
  test_dataloader = DataLoader(test_dataset,batch_size=batch_size,shuffle=False)

  return train_dataloader, test_dataloader
