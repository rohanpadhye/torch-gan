#!/usr/bin/python

import sys
import csv
import matplotlib.pyplot as plt

def main():
	xs = []
	ys = []
	csv_filename = sys.argv[1]
	plt_title = sys.argv[2]
	png_filename = sys.argv[3] if len(sys.argv) > 3 else None
	with open(csv_filename) as csv_file:
		reader = csv.reader(csv_file, delimiter=',')
		for row in reader:
			assert(len(row) == 2)
			xs.append(row[0])
			ys.append(row[1])
	
	style = 'bo' if csv_filename.endswith('in.csv') else 'ro'
	plt.plot(xs, ys, style)
	plt.title(plt_title)
	plt.xlabel('x1')
	plt.ylabel('x2')
	plt.axis([-1, 1, -1, 1])
	plt.grid(True)
	if png_filename is not None:
		plt.savefig(png_filename)
	else:
		plt.show()

if __name__ == "__main__":
	main()
