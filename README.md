# ResNet-CIFAR10-from-Scratch

This project reproduces and analyzes the key idea of ResNet, *"Deep Residual Learning for Image Recognition"*

## What I implemented

- Residual architecture with skip connection
- ResNet-20 / ResNet-44 / ResNet-56
- PlainNet-20 / PlainNet-44 / PlainNet-56
- CIFAR-10 training pipeline with data augmentation
- Memory complexity comparison using model summary
- Error-rate comparison
- Analysis of the degradation problem

## Key Result

PlainNet becomes harder to optimize as depth increases, namely meets *the degradation problem*, while ResNet benefits from residual connections and trains more stably.

## Experiments

| Experiment | Description |
|---|---|
| Memory complexity | Model summary comparison |
| ResNet44 vs PlainNet44 | Accuracy and training difficulty |
| ResNet20/56 vs PlainNet20/56 | Degradation problem |

## Tech Stack

PyTorch, torchvision, NumPy, Cuda, Matplotlib, Jupyter Notebook
