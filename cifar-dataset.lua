path = require 'pl.path'
require 'image'

Dataset = {}
local CIFAR, parent = torch.class("Dataset.CIFAR")

function get_Data(dataset, path, do_shuffling)
   local data = torch.Tensor(60000, 3, 32, 32)
   local label = torch.Tensor(60000)

   local train_data = torch.load(path..dataset..'-train.t7')
   data[{ {1, 50000} }] = train_data.data
   label[{ {1, 50000} }] = train_data.label

   local test_data = torch.load(path..dataset..'-test.t7')
   data[{ {50001, 60000} }] = test_data.data
   label[{ {50001, 60000} }] = test_data.label

   if do_shuffling then
      local shuffle = torch.randperm(50000)
      data[{ {1, 50000} }] = data:index(1, shuffle:long())
      label[{ {1, 50000} }] = label:index(1, shuffle:long())
   end
   -- raw labels were 0-based indexing, convert to 1-based
   return data, label + 1
end

function CIFAR:__init(data, label, mode, batchSize, augmentation)
   local trsize = 45000
   local vasize = 5000
   local tesize = 10000
   self.batchSize = batchSize
   self.mode = mode
   if mode == "train" then
      self.data = data[{ {1,trsize} }]
      self.label = label[{ {1,trsize} }]
      self.augmentation = augmentation
   elseif mode == "valid" then
      self.data = data[{ {trsize+1, trsize+vasize} }]
      self.label = label[{ {trsize+1, trsize+vasize} }]
   elseif mode == "test" then
      self.data = data[{ {trsize+vasize+1, trsize+vasize+tesize} }]
      self.label = label[{ {trsize+vasize+1, trsize+vasize+tesize} }]
   end
end

function CIFAR:preprocess(mean, std)
   mean = mean or self.data:mean(1)
   std = std or self.data:std()
   self.data:add(-mean:expandAs(self.data)):mul(1/std)
   return mean,std
end

function CIFAR:size()
   return self.data:size(1)
end

function CIFAR:sampleIndices(indices, batch)
   batch = batch or {inputs = torch.zeros(indices:size(1), 3, 32,32),
                     outputs = torch.zeros(indices:size(1))}
   if self.mode == "train" then
      if self.augmentation then
         batch.inputs:zero()
         for i,index in ipairs(torch.totable(indices)) do
            -- Copy self.data[index] into batch.inputs[i], with standard data augmentation
            local input = batch.inputs[i]
            input:zero()
            -- Translation by at most 4 pixels
            local xoffs, yoffs = torch.random(-4,4), torch.random(-4,4)
            local input_y = {math.max(1,   1 + yoffs),
                             math.min(32, 32 + yoffs)}
            local data_y = {math.max(1,   1 - yoffs),
                            math.min(32, 32 - yoffs)}
            local input_x = {math.max(1,   1 + xoffs),
                             math.min(32, 32 + xoffs)}
            local data_x = {math.max(1,   1 - xoffs),
                            math.min(32, 32 - xoffs)}
            input[{ {}, input_y, input_x }] = self.data[index][{ {}, data_y, data_x }]
            -- Horizontal flip, each side with half probability
            if torch.random(1,2)==1 then
               input:copy(image.hflip(input))
            end
         end
      else
         batch.inputs:copy(self.data:index(1, indices))
      end
   elseif self.mode=="test" or self.mode=="valid" then
      batch.inputs:copy(self.data:index(1, indices))
   end
   batch.outputs:copy(self.label:index(1, indices))
   return batch
end
