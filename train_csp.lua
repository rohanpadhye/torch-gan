------------------------------------------------------------
--- This code is based on the eyescream code released at
--- https://github.com/facebook/eyescream
--- If you find it usefull consider citing
--- http://arxiv.org/abs/1506.05751
------------------------------------------------------------

require 'hdf5'
require 'nngraph'
require 'nn'
require 'torch'
require 'cunn'
require 'optim'
require 'image'
require 'pl'
require 'paths'
ok, disp = pcall(require, 'display')
if not ok then print('display not found. unable to plot') end
adversarial = require 'lfw_adverserial'


----------------------------------------------------------------------
-- parse command-line options
opt = lapp[[
  -s,--save          (default "logs_csp")      subdirectory to save logs
  --saveFreq         (default 1)          save every saveFreq epochs
  -n,--network       (default "")          reload pretrained network
  -p,--plot                                plot while training
  -r,--learningRate  (default 0.001)        learning rate
  -b,--batchSize     (default 128)         batch size
  -m,--momentum      (default 0)           momentum, for SGD only
  --coefL1           (default 0)           L1 penalty on the weights
  --coefL2           (default 0)           L2 penalty on the weights
  -t,--threads       (default 4)           number of threads
  -g,--gpu           (default 0)           gpu to run on (default cpu)
  -d,--noiseDim      (default 8)           dimensionality of noise vector
  --K                (default 1)           number of iterations to optimize D for
  -w, --window       (default 3)           windsow id of sample image
  --dim              (default 2)           input dimensions
]]


if opt.gpu < 0 or opt.gpu > 3 then opt.gpu = false end

opt.hiddenD1 = opt.hiddenD1 or 16
opt.hiddenD2 = opt.hiddenD2 or 8
opt.noiseDim = opt.noiseDim or 8
opt.hiddenG1 = opt.hiddenG1 or 16
opt.hiddenG2 = opt.hiddenG2 or 64

print(opt)

-- fix seed
torch.manualSeed(1)

-- threads
torch.setnumthreads(opt.threads)
print('<torch> set nb of threads to ' .. torch.getnumthreads())

if opt.gpu then
  cutorch.setDevice(opt.gpu + 1)
  print('<gpu> using device ' .. opt.gpu)
  torch.setdefaulttensortype('torch.CudaTensor')
else
  torch.setdefaulttensortype('torch.FloatTensor')
end

opt.dim = opt.dim or 2
opt.geometry = {opt.dim, 1, 1}

local input_sz = opt.geometry[1] * opt.geometry[2] * opt.geometry[3]

if opt.network == '' then
  ----------------------------------------------------------------------
  -- define D network to train
  model_D = nn.Sequential()
  model_D:add(nn.Reshape(input_sz))
  model_D:add(nn.Linear(input_sz, opt.hiddenD1))
  model_D:add(nn.ReLU(true))
  model_D:add(nn.Linear(opt.hiddenD1, opt.hiddenD2))
  model_D:add(nn.ReLU(true))
  model_D:add(nn.Dropout())
  model_D:add(nn.Linear(opt.hiddenD2, 1))
  model_D:add(nn.Sigmoid())

  noise_input = nn.Identity()()
  lg = nn.Linear(opt.noiseDim, opt.hiddenG1)(noise_input)
  lg = nn.ReLU(true)(lg)
  lg = nn.Linear(opt.hiddenG1, opt.hiddenG2)(lg)
  lg = nn.ReLU(true)(lg)
  lg = nn.Linear(opt.hiddenG2, input_sz)(lg)
  lg = nn.Tanh()(lg)
  model_G = nn.gModule({noise_input}, {lg})

  model_G = nn.Sequential()
  model_G:add(nn.Linear(opt.noiseDim, opt.hiddenG1))
  model_G:add(nn.ReLU(true))
  model_G:add(nn.Linear(opt.hiddenG1, opt.hiddenG2))
  model_G:add(nn.ReLU(true))
  model_G:add(nn.Linear(opt.hiddenG2, input_sz))
  model_G:add(nn.Tanh())
  model_G:add(nn.Reshape(opt.geometry[1], opt.geometry[2], opt.geometry[3]))

else
  print('<trainer> reloading previously trained network: ' .. opt.network)
  tmp = torch.load(opt.network)
  model_D = tmp.D
  model_G = tmp.G
end

-- loss function: negative log-likelihood
criterion = nn.BCECriterion()

-- retrieve parameters and gradients
parameters_D,gradParameters_D = model_D:getParameters()
parameters_G,gradParameters_G = model_G:getParameters()

-- print networks
print('Discriminator network:')
print(model_D)
print('Generator network:')
print(model_G)

-- load data
local cspHd5 = hdf5.open('datasets/csp.h5', 'r')
local data = cspHd5:read('csp'):all()
cspHd5:close()
print('Data size: ')
print(data:size())

ndata = data:size()[1]
ntrain = math.floor(ndata * 0.9)
trainData = data[{{1, ntrain}}]
valData = data[{{ntrain+1, ndata}}]
print(ndata .. ' samples loaded, ' .. trainData:size()[1] .. ' training and ' .. valData:size()[1] .. ' validation')


-- this matrix records the current confusion across classes
classes = {'0','1'}
confusion = optim.ConfusionMatrix(classes)

-- log results to files
trainLogger = optim.Logger(paths.concat(opt.save, 'train.log'))
testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))

if opt.gpu then
  print('Copy model to gpu')
  model_D:cuda()
  model_G:cuda()
end

-- Training parameters
sgdState_D = {
  learningRate = opt.learningRate,
  momentum = opt.momentum,
  optimize=true,
  numUpdates = 0
}

sgdState_G = {
  learningRate = opt.learningRate,
  momentum = opt.momentum,
  optimize=true,
  numUpdates=0
}

-- Get examples to plot
function getSamples(dataset, N)
  local numperclass = numperclass or 10
  local N = N or 8
  local noise_inputs = torch.Tensor(N, opt.noiseDim)

  -- Generate samples
  noise_inputs:normal(0, 1)
  local samples = model_G:forward(noise_inputs)
  samples = nn.HardTanh():forward(samples) -- FIXME: is the hardtanh() required?
  local to_plot = {}
  for i=1,N do
    to_plot[#to_plot+1] = samples[i]:float():reshape(opt.dim)
  end

  return to_plot
end

function getTrainSamples(N) 
  local to_plot = {}
  for i=1,N do
    to_plot[#to_plot+1] = trainData[i]:clone():float():reshape(opt.dim)
  end
  return to_plot
end

	

function plotSamples(samples, filename)
	local out = assert(io.open(filename, "w"))
	for i, x in ipairs(samples) do
		for j = 1, (opt.dim-1) do
			out:write(x[j] .. ", ")
		end
		out:write(x[opt.dim] .. "\n")
	end
	out:close()
end

plotSamples(getTrainSamples(50), paths.concat(opt.save, "in.csv"))

-- training loop
local k = 1
while true do
  local to_plot = getSamples(valData, 50)
  torch.setdefaulttensortype('torch.FloatTensor')

  trainLogger:style{['% mean class accuracy (train set)'] = '-'}
  testLogger:style{['% mean class accuracy (test set)'] = '-'}
  trainLogger:plot()
  testLogger:plot()

  
  plotSamples(to_plot, paths.concat(opt.save, "gen-"..k..".csv"))
  k = k + 1

  if opt.gpu then
    torch.setdefaulttensortype('torch.CudaTensor')
  else
    torch.setdefaulttensortype('torch.FloatTensor')
  end


  -- train/test
  adversarial.train(trainData)
  adversarial.test(valData)

  sgdState_D.momentum = math.min(sgdState_D.momentum + 0.0008, 0.7)
  sgdState_D.learningRate = math.max(opt.learningRate*0.99^epoch, 0.000001)
  sgdState_G.momentum = math.min(sgdState_G.momentum + 0.0008, 0.7)
  sgdState_G.learningRate = math.max(opt.learningRate*0.99^epoch, 0.000001)


end
