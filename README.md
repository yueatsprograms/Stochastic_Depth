Deep Networks with Stochastic Depth
====================
This repository hosts the Torch 7 code for the paper _Deep Networks with Stochastic Depth_
available at http://arxiv.org/abs/1603.09382. For now, the code reproduces the results in Figure 3 for CIFAR-10 and CIFAR-100, and Figure 4 left for SVHN. The code for the 1202-layer network is easily modified from the repo `fb.resnet.torch` using our provided module for stochastic depth.

### Table of Contents
- [Updates](#updates)  
- [Prerequisites](#prerequisites)  
- [Getting Started on CIFAR-10](#getting-started-on-cifar-10)  
- [Usage Details](#usage-details)  
- [Known Problems](#known-problems) 
- [Contact](#contact)  

### Updates
Please see the [latest implementation](https://github.com/felixgwu/img_classification_pk_pytorch) of stochastic depth and other cool models (DenseNet etc.) in PyTorch, by Felix Wu and Danlu Chen. Their code is much more memory efficient, more user friendly and better maintained. The 1202-layer architecture on CIFAR-10 can be trained on one TITAN X (amazingly!) under our standard settings.

### Prerequisites
- Torch 7 and CUDA with the basic packages (nn, optim, image, cutorch, cunn).
- [cudnn](https://developer.nvidia.com/cudnn) and [torch bindings](https://github.com/soumith/cudnn.torch).
- [nninit](https://github.com/Kaixhin/nninit); `luarocks install nninit` should do the trick.
- CIFAR-10 and CIFAR-100 datasets in Torch format; [this script](https://github.com/soumith/cifar.torch) should very conveniently handle it for you.
- SVHN dataset in Torch format, available [here](http://torch7.s3-website-us-east-1.amazonaws.com/data/svhn.t7.tgz). Please note that running on SVHN requires roughly 28GB of RAM for dataset loading.

### Getting Started on CIFAR-10
```bash
git clone https://github.com/yueatsprograms/Stochastic_Depth
cd Stochastic_Depth
git clone https://github.com/soumith/cifar.torch
cd cifar.torch
th Cifar10BinToTensor.lua
cd ..
mkdir results
th main.lua -dataRoot cifar.torch/ -resultFolder results/ -deathRate 0.5
```

### Usage Details
`th main.lua -dataRoot path_to_data -resultFolder path_to_save -deathRate 0.5`<br/>
This command runs the 110-layer ResNet on CIFAR-10 with stochastic depth, using _linear decay_ survival probabilities ending in 0.5. The `-device` flag allows you to specify which GPU to run on. On our machine with a TITAN X, each epoch takes about 60 seconds, and the program ends with a test error (selected by best validation error) of __5.25%__.

The default deathRate is set to 0. This is equivalent to a constant depth network, so to run our baseline, enter: <br/>
`th main.lua -dataRoot path_to_data -resultFolder path_to_save` <br/>
On our machine with a TITAN X, each epoch takes about 75 seconds, and this baseline program ends with a test error (selected by best validation error) of 6.41% (see Figure 3 in the paper).

You can run on CIFAR-100 by adding the flag `-dataset cifar100`. Our program provides other options, for example, your network depth (`-N`), data augmentation (`-augmentation`), batch size (`-batchSize`) etc. You can change the optimization hyperparameters in the sgdState variable, and learning rate schedule in the main function. The program saves a file every epoch to `resultFolder`/errors\_`N`\_`dataset`\_`deathMode`\_`deathRate`, which has a table of tuples containing your test and validation errors until that epoch.

The architecture and number of epochs for SVHN used in our paper are slightly different from the code's default, please use the following command if you would like to replicate our result of 1.75% on SVHN:<br/>
`th main.lua -dataRoot path_to_data -resultFolder path_to_save -dataset svhn -N 25 -maxEpochs 50 -deathRate 0.5`

### Known Problems
- It is normal to get a +/- 0.2% difference from our reported results on CIFAR-10, and analogously for the other datasets. Networks are initialized differently, and most importantly, the validation set is chosen at random (determined by your seed).
- If you train on SVHN and the model doesn't converge for the first 1600 or so iterations, that's ok, just wait for a little longer.
- <a href="https://github.com/xgastaldi"> Xavier <a/> reported that the model is able to converge for him on CIFAR-10 only after he uses the following initalization for Batch Normalization `model:add(cudnn.SpatialBatchNormalization(_dim_):init('weight', nninit.normal, 1.0, 0.002):init('bias', nninit.constant, 0))`. We could not replicate the non-convergence and thus won't put this initialization into our code, but recognize that machines (or the versions of Torch installed) might be different.

### Contact
My email is ys646 at cornell.edu. I'm happy to answer any of your questions, and I'd very much appreciate your suggestions. My academic website is at http://yueatsprograms.github.io.