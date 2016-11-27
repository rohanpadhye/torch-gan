import random, math

def noise():
	r = random.random()
	return 2*r - 1.0

for i in range(1000):
	z_1 = noise()
	z_2 = noise()
	r = 0.5 * math.sqrt(abs(z_1))
	t = z_2 * math.pi
	x_1 = r * math.cos(t)
	x_2 = r * math.sin(t)
	print str(x_1) + ',' + str(x_2)
