path = require 'pl.path'
require 'image'
require 'nn'

Dataset = {}
local SVHN, parent = torch.class("Dataset.LOADER")

function get_Data(dataset, path, do_shuffling)
	assert(dataset == 'svhn')
	print ('=================== loading dataset ===================')
	local size_train = 73257
	local size_test = 26032
	local size_extra = 531131
	local data = torch.Tensor(size_train + size_extra + size_test, 3, 32, 32)
	local label = torch.Tensor(size_train + size_extra + size_test)
	local subset = torch.load(path..'train_32x32.t7','ascii')
	data[{ {1, size_train} }] = subset.X
	label[{ {1, size_train} }] = subset.y
   	subset = torch.load(path..'extra_32x32.t7','ascii')
   	data[{ {size_train + 1, size_train + size_extra} }] = subset.X
   	label[{ {size_train + 1, size_train + size_extra} }] = subset.y
	subset = torch.load(path..'test_32x32.t7','ascii')
	data[{ {size_train + size_extra + 1, size_train + size_extra + size_test} }] = subset.X
	label[{ {size_train + size_extra + 1, size_train + size_extra + size_test} }] = subset.y
	
	if do_shuffling then
		print ('=================== data shuffling  ===================')
		local shuffle = torch.randperm(size_train)
		data[{ {1, size_train} }] = data:index(1, shuffle:long())
		label[{ {1, size_train} }] = label:index(1, shuffle:long())
		local shuffle_extra = torch.randperm(size_extra) + size_train
		data[{ {size_train + 1, size_train + size_extra} }] = data:index(1, shuffle_extra:long())
		label[{ {size_train + 1, size_train + size_extra} }] = label:index(1, shuffle_extra:long())
	end
	return data, label
end

function SVHN:__init(data, label, mode)
	local vasize = 10 * ( 400 + 200 ) 
	local trsize = 73257 + 531131 - vasize
	local tesize = 26032
	local idx_val = torch.Tensor(vasize)
	local idx_tr = torch.Tensor(trsize)
	local n_tr = 0
	for i = 1, 10 do
		idx_i = label[{{1,73257}}]:eq(i):nonzero()
		n_i = idx_i:size(1)
		idx_val[{{ (i-1) * 400 + 1, i * 400 }}] = idx_i[{{ 1,400 }}]		
		idx_tr[{{ n_tr + 1, n_tr + n_i -400}}] = idx_i[{{ 401, n_i}}]
		n_tr = n_tr + n_i -400
	end
	for i = 1, 10 do 		-- for the extra training set
		idx_i = label[{{73258,604388}}]:eq(i):nonzero()
		n_i = idx_i:size(1)
		idx_val[{{ 4001 + (i-1) * 200, 4000 + i * 200 }}] = idx_i[{{ 1,200 }}] + 73257
		idx_tr[{{ n_tr + 1, n_tr + n_i -200}}] = idx_i[{{ 201, n_i}}] + 73257 
		n_tr = n_tr + n_i -200
    end

	self.mode = mode
	if mode == "train" then
	  self.data = data:index(1, idx_tr:long())
	  self.label = label:index(1, idx_tr:long())
	elseif mode == "valid" then
	  self.data = data:index(1, idx_val:long())
	  self.label = label:index(1, idx_val:long())
	elseif mode == "test" then
	  self.data = data[{ {trsize+vasize+1, trsize+vasize+tesize} }]
	  self.label = label[{ {trsize+vasize+1, trsize+vasize+tesize} }]
	end
end

function SVHN:preprocess(mean, std)
   mean = mean or self.data:mean(1)
   std = std or self.data:std()
   self.data:add(-mean:expandAs(self.data)):mul(1/std)
   return mean,std
end

function SVHN:size()
	return self.data:size(1)
end

function SVHN:sampleIndices(indices, batch)
	batch = batch or {inputs = torch.zeros(indices:size(1), 3, 32,32),
	                 outputs = torch.zeros(indices:size(1))}
	batch.inputs:copy(self.data:index(1, indices))
	batch.outputs:copy(self.label:index(1, indices))
	return batch
end
