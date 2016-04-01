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
`th main.lua -dataRoot path_to_data -resultFolder path_to_save -deathRate 0.5`<br/>
This command runs the 110-layer ResNet on CIFAR-10 with stochastic depth, using linear decaying survival probabilities ending in 0.5. The `-device` flag allows you to specify which GPU to run on. On our machine with a TITAN X, each epoch takes about 60 seconds, and the program ends with a test error (selected by best validation error) of __5.23%__.

The default deathRate is set to 0. This is equivalent to a constant depth network, so to run our baseline, enter: <br/>
`th main.lua -dataRoot path_to_data -resultFolder path_to_save` <br/>
On our machine with a TITAN X, each epoch takes about 75 seconds, and this baseline program ends with a test error (selected by best validation error) of 6.41% (see Figure 3 in the paper).

You can run on CIFAR-100 by adding the flag `-dataset cifar100`. Our program provides other options, for example, your network depth (`-N`), data augmentation (`-augmentation`), batch size (`-batchSize`) etc. You can change the optimization hyperparameters in the sgdState variable, and learning rate schedule in the the main function. The program saves a file to `resultFolder`/errors\_`N`\_`dataset`\_`deathMode`\_`deathRate`, which is a table of `nEpochs` many tuples, each containing your test and validation error at the end of that epoch.

### Contact
My email is ys646 at cornell.edu. I'm happy to answer any of your question, and I'd very much appreciate your suggestions. 