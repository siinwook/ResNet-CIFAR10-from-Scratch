# ResNet-CIFAR10-from-Scratch

This project reproduces and analyzes the key idea of ResNet, *"Deep Residual Learning for Image Recognition"*

## What I implemented

- Residual architecture with skip connection
- ResNet-20 / ResNet-44 / ResNet-56
- PlainNet-20 / PlainNet-44 / PlainNet-56
- CIFAR-10 training pipeline with data augmentation
- Memory complexity comparison using model summary
- Error-rate comparison
- Analysis of *the degradation problem*

## Key Result

- Residual architectures address *the degradation problem* and make deeper neural networks easier to optimize.
- Comparing shallow models, ResNet gains much accuracy and converges faster than PlainNet.
- Comparing deeper models, PlainNet accuracy saturated while ResNet benefits the deep layer

## Experiments

| Experiment | Description |
|---|---|
| Memory complexity | Model summary comparison |
| ResNet44 vs PlainNet44 | Accuracy and training difficulty |
| ResNet20/56 vs PlainNet20/56 | *The degradation problem* |

## How to Run

```bash
git clone https://github.com/siinwook/ResNet-CIFAR10-from-Scratch.git
cd ResNet-CIFAR10-from-Scratch

pip install -r requirements.txt
python -m src.train
```
## Tech Stack

PyTorch, torchvision, NumPy, Cuda, Matplotlib, Jupyter Notebook
