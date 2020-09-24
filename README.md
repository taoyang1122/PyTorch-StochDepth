# PyTorch-StochDepth
This is an unofficial PyTorch implementation for the Stochastic Depth method (https://arxiv.org/abs/1603.09382). The official Torch implementation is https://github.com/yueatsprograms/Stochastic_Depth. 

# Run
```shell
python train.py  \
  --cifar-type 100  \ # cifar100 or cifar10 
```

# Results
Below is the Top-1 accuracy reported in the paper and reproduced by this repo.
|ResNet-110|Cifar-10|Cifar-100|
|----------|--------|---------|
|Baseline|93.59|72.24|
|StochDepth-reported|94.75|75.02|
|StochDepth-reproduced|94.29|75.52|

# Reference
\- https://github.com/shamangary/Pytorch-Stochastic-Depth-Resnet
