#!/usr/bin/python

import sys
import csv

def circle(x):
	return (x[0]**2 + x[1]**2) < 0.5

f = circle

def main():
	total = 0
	good = 0
	csv_filename = sys.argv[1]
	with open(csv_filename) as csv_file:
		reader = csv.reader(csv_file, delimiter=',')
		for row in reader:
			assert(len(row) == 2)
			x = [float(row[0]), float(row[1])]
			total += 1
			if f(x):
				good += 1

	success = float(good)/float(total)*100.0
	print "Success = " + str(success) + "%"

if __name__ == "__main__":
	main()
