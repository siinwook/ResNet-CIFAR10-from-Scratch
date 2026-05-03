# ResNet on CIFAR-10 from Scratch

This project reproduces and analyzes the key idea of ResNet: residual learning and the degradation problem.

## What I implemented

- ResNet-20 / ResNet-44 / ResNet-56
- PlainNet-20 / PlainNet-44 / PlainNet-56
- CIFAR-10 training pipeline with data augmentation
- Accuracy and error-rate comparison
- Memory complexity comparison using model summary
- Analysis of the degradation problem

## Key Result

PlainNet becomes harder to optimize as depth increases, while ResNet benefits from residual connections and trains more stably.

## Experiments

| Experiment | Description |
|---|---|
| ResNet44 vs PlainNet44 | Accuracy and training difficulty |
| ResNet20/56 vs PlainNet20/56 | Degradation problem |
| Memory complexity | Model summary comparison |

## Tech Stack

PyTorch, torchvision, NumPy, Matplotlib, Jupyter Notebook
