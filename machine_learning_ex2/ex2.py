from machine_learning_ex2.plot_data import plot_data
from machine_learning_ex2.cost_function import cost_function
import numpy as np

if __name__ == '__main__':

	data = np.loadtxt('ex2data1.txt', delimiter=',')
	x = data[:,:2]
	y = np.c_[data[:,2]]

	"""
		Part 1: Plotting
	"""
	# We start the exercise by first plotting the data to understand the
	# the problem we are working with

	print('Plotting data with + indicating (y = 1) examples and o indicating (y = 0) examples.')
	plot_data()

	"""
		Part 2: Compute Cost and Gradient
		
		In this part of the exercise, you will implement the cost and gradient
		for logistic regression. You neeed to complete the code in cost_function
	"""

	# Setup the data matrix appropriately, and add ones for the intercept term
	[m, n] = x.shape

	# Add intercept term to x and x_test
	x = np.c_[np.ones((m,1)), x]

	# Initializing fitting parameters
	initial_theta = np.zeros((n + 1, 1))