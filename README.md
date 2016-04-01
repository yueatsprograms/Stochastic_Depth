Deep Networks with Stochastic Depth
====================
This repository hosts the Torch 7 code for the paper _Deep Networks with Stochastic Depth_
available at http://arxiv.org/abs/1603.09382v1. For now, the code reproduces the results in Figure 3 on CIFAR-10 and CIFAR-100. The code for SVHN and 1202-layer CIFAR-10 (which requires some memory optimization) will be available very soon.

### Prerequisites
- Torch 7 and CUDA with the basic packages (nn, optim, cutorch, cunn)
- cudnn (https://developer.nvidia.com/cudnn) and torch bindings (https://github.com/soumith/cudnn.torch)
- nninit torch package (https://github.com/Kaixhin/nninit)
- CIFAR-10 and CIFAR-100 dataset in Torch format, this script https://github.com/soumith/cifar.torch should very conveniently handle it for you

### Getting Started
This command runs the 110-layer ResNet on CIFAR-10 with stochastic depth, using linear decaying survival probabilities ending in 0.5 <br/>
`th main.lua -dataRoot path_to_data -resultFolder path_to_save -N 18 -deathRate 0.5`<br/>
the `-device` flag allows you to specify which GPU to run on. <br/>
Setting deathRate to 0 is equivalent to a constant depth network, so to run our baseline, enter
`th main.lua -dataRoot path_to_data -resultFolder path_to_save -N 18` <br/>
You can run on CIFAR-100 by adding the flag `-dataset cifar100`.

### Contact
My email is ys646 at cornell.edu. I'm happy to answer any of your question, and I'd very much appreciate your suggestions. 
