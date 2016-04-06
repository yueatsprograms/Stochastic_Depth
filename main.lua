require 'nn'
require 'cutorch'
require 'cunn'
require 'cudnn'
require 'optim'
local nninit = require 'nninit'

require 'ResidualDrop'

-- Saves 40% time according to http://torch.ch/blog/2016/02/04/resnets.html
cudnn.fastest = true
cudnn.benchmark = true

opt = lapp[[
  --maxEpochs     (default 500)         Maximum number of epochs to train the network
  --batchSize     (default 128)         Mini-batch size
  --N             (default 18)          Model has 6*N+2 convolutional layers
  --dataset       (default cifar10)     Use cifar10, cifar100 or svhn
  --deathMode     (default lin_decay)   Use lin_decay or uniform
  --deathRate     (default 0)           1-p_L for lin_decay, 1-p_l for uniform, 0 is constant depth
  --device        (default 0)           Which GPU to run on, 0-based indexing
  --augmentation  (default true)        Standard data augmentation (CIFAR only), true or false 
  --resultFolder  (default "")          Path to the folder where you'd like to save results
  --dataRoot      (default "")          Path to data (e.g. contains cifar10-train.t7)
]]
print(opt)
cutorch.setDevice(opt.device+1)   -- torch uses 1-based indexing for GPU, so +1
cutorch.manualSeed(1)
torch.manualSeed(1)
torch.setnumthreads(1)            -- number of OpenMP threads, 1 is enough

---- Loading data ----
if opt.dataset == 'svhn' then require 'svhn-dataset' else require 'cifar-dataset' end
all_data, all_labels = get_Data(opt.dataset, opt.dataRoot, true)  -- default do shuffling
dataTrain = Dataset.LOADER(all_data, all_labels, "train", opt.batchSize, opt.augmentation)
dataValid = Dataset.LOADER(all_data, all_labels, "valid", opt.batchSize)
dataTest = Dataset.LOADER(all_data, all_labels, "test", opt.batchSize)
local mean,std = dataTrain:preprocess()
dataValid:preprocess(mean,std)
dataTest:preprocess(mean,std)
print("Training set size:\t",   dataTrain:size())
print("Validation set size:\t", dataValid:size())
print("Test set size:\t\t",     dataTest:size())

---- Optimization hyperparameters ----
sgdState = {
   weightDecay   = 1e-4,
   momentum      = 0.9,
   dampening     = 0,
   nesterov      = true,
}
-- Point at which learning rate decrease by 10x
lrSchedule = {svhn     = {0.6, 0.7 }, 
              cifar10  = {0.5, 0.75},
              cifar100 = {0.5, 0.75}}

---- Buidling the residual network model ----
-- Input: 3x32x32
print('Building model...')
model = nn.Sequential()
------> 3, 32,32
model:add(cudnn.SpatialConvolution(3, 16, 3,3, 1,1, 1,1)
            :init('weight', nninit.kaiming, {gain = 'relu'})
            :init('bias', nninit.constant, 0))
model:add(cudnn.SpatialBatchNormalization(16))--:init('weight', nninit.normal, 1.0, 0.002)
                                              --:init('bias', nninit.constant, 0))
model:add(cudnn.ReLU(true))
------> 16, 32,32   First Group
for i=1,opt.N do   addResidualDrop(model, nil, 16)   end
------> 32, 16,16   Second Group
addResidualDrop(model, nil, 16, 32, 2)
for i=1,opt.N-1 do   addResidualDrop(model, nil, 32)   end
------> 64, 8,8     Third Group
addResidualDrop(model, nil, 32, 64, 2)
for i=1,opt.N-1 do   addResidualDrop(model, nil, 64)   end
------> 10, 8,8     Pooling, Linear, Softmax
model:add(nn.SpatialAveragePooling(8,8)):add(nn.Reshape(64))
if opt.dataset == 'cifar10' or opt.dataset == 'svhn' then
  model:add(nn.Linear(64, 10))
elseif opt.dataset == 'cifar100' then
  model:add(nn.Linear(64, 100))
else
  print('Invalid argument for dataset!')
end
model:add(cudnn.LogSoftMax())
model:cuda()

loss = nn.ClassNLLCriterion()
loss:cuda()
collectgarbage()
-- print(model)   -- if you need to see the architecture, it's going to be long!

---- Determines the position of all the residual blocks ----
addtables = {}
for i=1,model:size() do
    if tostring(model:get(i)) == 'nn.ResidualDrop' then addtables[#addtables+1] = i end
end

---- Sets the deathRate (1 - survival probability) for all residual blocks  ----
for i,block in ipairs(addtables) do
  if opt.deathMode == 'uniform' then
    model:get(block).deathRate = opt.deathRate
  elseif opt.deathMode == 'lin_decay' then
    model:get(block).deathRate = i / #addtables * opt.deathRate
  else
    print('Invalid argument for deathMode!')
  end
end

---- Resets all gates to open ----
function openAllGates()
  for i,block in ipairs(addtables) do model:get(block).gate = true end
end

---- Testing ----
function evalModel(dataset)
  model:evaluate()
  openAllGates() -- this is actually redundant, test mode never skips any layer
  local correct = 0
  local total = 0
  local batches = torch.range(1, dataset:size()):long():split(opt.batchSize)
  for i=1,#batches do
     local batch = dataset:sampleIndices(batches[i])
     local inputs, labels = batch.inputs, batch.outputs:long()
     local y = model:forward(inputs:cuda()):float()
     local _, indices = torch.sort(y, 2, true)
     -- indices is a tensor with shape (batchSize, nClasses)
     local top1 = indices:select(2, 1)
     correct = correct + torch.eq(top1, labels):sum()
     total = total + indices:size(1)
  end
  return 1-correct/total
end

-- Saving and printing results
all_results = {}  -- contains test and validation error throughout training
-- For CIFAR, accounting is done every epoch, and for SVHN, every 200 iterations
function accounting(training_time)
  local results = {evalModel(dataValid), evalModel(dataTest)}
  all_results[#all_results + 1] = results
  -- Saves the errors. These get covered up by new ones every time the function is called
  torch.save(opt.resultFolder .. string.format('errors_%d_%s_%s_%.1f', 
    opt.N, opt.dataset, opt.deathMode, opt.deathRate), all_results)
  if opt.dataset == 'svhn' then 
    print(string.format('Iter %d:\t%.2f%%\t\t%.2f%%\t\t%0.0fs', 
      sgdState.iterCounter, results[1]*100, results[2]*100, training_time))
  else
    print(string.format('Epoch %d:\t%.2f%%\t\t%.2f%%\t\t%0.0fs', 
      sgdState.epochCounter, results[1]*100, results[2]*100, training_time))
  end
end

---- Training ----
function main()  
  local weights, gradients = model:getParameters()
  sgdState.epochCounter  = 1
  if opt.dataset == 'svhn' then 
    sgdState.iterCounter = 1 
    print('Training...\nIter\t\tValid. err\tTest err\tTraining time')
  else
    print('Training...\nEpoch\tValid. err\tTest err\tTraining time')
  end
  local all_indices = torch.range(1, dataTrain:size())
  local timer = torch.Timer()
  while sgdState.epochCounter <= opt.maxEpochs do
    -- Learning rate schedule
    if sgdState.epochCounter < opt.maxEpochs*lrSchedule[opt.dataset][1] then
      sgdState.learningRate = 0.1
    elseif sgdState.epochCounter < opt.maxEpochs*lrSchedule[opt.dataset][2] then
      sgdState.learningRate = 0.01
    else
      sgdState.learningRate = 0.001
    end

    local shuffle = torch.randperm(dataTrain:size())
    local batches = all_indices:index(1, shuffle:long()):long():split(opt.batchSize)
    for i=1,#batches do
        model:training()
        openAllGates()    -- resets all gates to open
        -- Randomly determines the gates to close, according to their survival probabilities
        for i,tb in ipairs(addtables) do
          if torch.rand(1)[1] < model:get(tb).deathRate then model:get(tb).gate = false end
        end
        function feval(x)
            gradients:zero()
            local batch = dataTrain:sampleIndices(batches[i])
            local inputs, labels = batch.inputs, batch.outputs:long()
            inputs = inputs:cuda()
            labels = labels:cuda()
            local y = model:forward(inputs)
            local loss_val = loss:forward(y, labels)
            local dl_df = loss:backward(y, labels)
            model:backward(inputs, dl_df)
            return loss_val, gradients
        end
        optim.sgd(feval, weights, sgdState)
        if opt.dataset == 'svhn' then
          if sgdState.iterCounter % 200 == 0 then
            accounting(timer:time().real)
            timer:reset()
          end
          sgdState.iterCounter = sgdState.iterCounter + 1
        end
    end
    if opt.dataset ~= 'svhn' then
      accounting(timer:time().real)
      timer:reset()
    end    
    sgdState.epochCounter = sgdState.epochCounter + 1
  end
  -- Saves the the last model, optional. Model loading feature is not available now but is easy to add
  -- torch.save(opt.resultFolder .. string.format('model_%d_%s_%s_%.1f', 
  --    opt.N, opt.dataset, opt.deathMode, opt.deathRate), model)
end

main()
