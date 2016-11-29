#!/usr/bin/python

import sys
import csv
import math
import numpy as np

def circle(x_1, x_2):
	return (x_1**2 + x_2**2) < 0.5

def rectangle(x_1, x_2):
   return x_1 > 0.25 and x_1 < 0.75 and x_2 < -0.1 and x_2 > -0.8

def ring(x_1, x_2):
  return (x_1 * x_1 + x_2 * x_2 < 0.5) and (x_1 * x_1 + x_2 * x_2 > 0.2)

def bands(x_1, x_2):
  return math.sin(4 * (x_1 + x_2)) > 0.4


f = ring



MIN =    -1.0
MAX =     1.0
DELTA =  0.01
DIVISIONS = int((MAX - MIN)/DELTA)
grid = np.zeros((DIVISIONS,DIVISIONS), np.int8)

def main():
	total = 0
	good = 0
	csv_filename = sys.argv[1]
	with open(csv_filename) as csv_file:
		reader = csv.reader(csv_file, delimiter=',')
		for row in reader:
			assert(len(row) == 2)
			x_1 = float(row[0])
			x_2 = float(row[1])
			total += 1
			if f(x_1, x_2):
				good += 1
			inc_count(x_1, x_2)

	success = float(good)/float(total)*100.0
	print "Success = " + str(success) + "%"
	compute_density_var(total)



def inc_count(x_1, x_2):
	i = int((x_1 - MIN)/DELTA)
	j = int((x_2 - MIN)/DELTA)
	grid[i][j] += 1

def grid_cell_in_region(x_1, x_2):
	return f(x_1, x_2) and f(x_1 + DELTA, x_2) and f(x_1, x_2 + DELTA) and f(x_1 + DELTA, x_2 + DELTA)

def compute_density_var(sample_count):
	grid_cells_in_region = 0
	counts = [];
	for x_1 in np.arange(MIN, MAX, DELTA):
		for x_2 in np.arange(MIN, MAX, DELTA):
			if grid_cell_in_region(x_1, x_2):
				i = int((x_1 - MIN)/DELTA)
				j = int((x_2 - MIN)/DELTA)
				counts.append(grid[i][j])
				grid_cells_in_region += 1
	sum_counts = float(sum(counts))
	max_entropy = -math.log(1/sum_counts) if sum_counts > 0 else 0.0
	fractions = [count / sum_counts for count in counts]
	entropy = -sum([p * math.log(p) for p in fractions if p > 0])
	relative_entropy = entropy / max_entropy
	print "Region density (" + str(DIVISIONS*DIVISIONS) + \
		" cells) : relative_entropy = " + str(relative_entropy)
	 


if __name__ == "__main__":
	main()
