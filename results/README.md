# Experimental Results

This document summarizes the main experimental results and my observations.
Training conditions of 2,3 are indicated below)

---

## 1. Memory Complexity

![Memory Complexity](./memory_complexity_summary.png)

### Observation

- ResNet44 and PlainNet44 have almost similar parameter counts.

### Interpretation

- ResNet44 requires marginally more parameters due to shortcut operations(Option B according to this paper), but the difference is negligible.

---

## 2. ResNet44 vs PlainNet44 Error Rate

![ResNet44 vs PlainNet44 Accuracy](./resnet44_vs_plainnet44_accuracy.png)

### Observation

- ResNet44 gains higher accuracy(by 9.0%) than PlainNet44
- ResNet44 starts convergence earlier than PlainNet44

### Interpretation

- Deep neural network with residual architecture is easier to optimize
- Skip connections used in every residual blocks, which represented as identity mapping, helps the responses flow easier.  

---

## 3. Degradation Problem: 20-layer vs 56-layer Networks

![Degradation Problem](./degradation_problem_error.png)

### Observation

- Plain network gains seriously lower accuracy when layer gets deeper(20 -> 56 depth).
- Residual network even gains slightly more accuracy when layer gets deeper(20 -> 56 depth).
- Considering shallower depth(20-depth), ResNet20 converges faster than PlainNet20 and gains more accuracy(by 2.0%).
- Considering deeper depth(56-depth), ResNet56 converges much faster than PlainNet56 and gains a far more accuracy(by 16.0%).  

### Interpretation

- Plain network suffers from *the degradation problem*(accuracy satured at some specific depth and decrease seriously), this phenomena occurs due to neither overfitting(no train/test accuracy gap) nor vanishing gradients(addressed by BN)
- Residual network addresses *the degradation problem* .

---

**Training conditions**

|||
|---|---|
| Dataset | CIFAR-10 |
| Optimizer | Adam(1-35 epoch) / SGD(36-50 epoch) |
| Learning rate | 0.001(1-20 epoch) / 0.0001(21-50 epoch) |
| Batch size | 128 |
| Epochs | 50 |
| Data augmentation | Random crop, Horizontal flip, Color jitter, Gray scale |
| Weight decay | 0.001 |
| Device: cuda |
