#!/usr/bin/python

import sys
import csv

def circle(x_1, x_2):
	return (x_1**2 + x_2**2) < 0.5

def rectangle(x_1, x_2):
   return x_1 > 0.25 and x_1 < 0.75 and x_2 < -0.1 and x_2 > -0.8

f = circle

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

	success = float(good)/float(total)*100.0
	print "Success = " + str(success) + "%"

if __name__ == "__main__":
	main()
