require 'hdf5'


local data = torch.rand(15000, 2, 1, 1):mul(0.1):add(0.4)


local cspHd5 = hdf5.open('datasets/csp.h5', 'w')
cspHd5:write('/csp', data)
cspHd5:close()
