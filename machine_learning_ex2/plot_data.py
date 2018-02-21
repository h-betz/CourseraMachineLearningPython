import numpy as np
import matplotlib.pyplot as plt

"""
	Plots the data points X and y into a new figure 
	PLOTDATA(x,y) plots the data points with + for the positive examples
	and o for the negative examples. X is assumed to be a Mx2 matrix.
"""
def plot_data(x, y):
	# Plot the positive and negative examples on a 2D plot, using the option 'k+' for the positive
	# examples and 'ko' for the negative examples.
	pos = list(filter(lambda i: i == 1, y))
	neg = list(filter(lambda i: i == 0, y))

	# TODO figure out the plotting
	plt.plot(x[pos,1], x[pos,2])