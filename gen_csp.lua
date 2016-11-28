require 'hdf5'

local N = 10000

local data = torch.rand(N, 2, 1, 1)

function circle(x_1, x_2) 
	return torch.pow(x_1, 2) + torch.pow(x_2, 2) < 0.5
end

function ring(x_1, x_2) 
	return (x_1 * x_1 + x_2 * x_2 < 0.5) and (x_1 * x_1 + x_2 * x_2 > 0.2)
end

function rectangle(x_1, x_2)
	return x_1 > 0.25 and x_1 < 0.75 and x_2 < -0.1 and x_2 > -0.8
end

function bands(x_1, x_2)
	return torch.sin(4 * (x_1 + x_2)) > 0.4
end


local f = ring

local i = 1
local attempts = 1
while i <= N do
	local x = data[i]
	x:uniform(-1, 1)
	attempts = attempts + 1
	local x_1 = x[1][1][1]
	local x_2 = x[2][1][1]
	if f(x_1, x_2) then
		i = i + 1
	end
end

success=N/attempts*100.0
print ("Success = " .. success .. "%")

local cspHd5 = hdf5.open('datasets/csp.h5', 'w')
cspHd5:write('/csp', data)
cspHd5:close()
