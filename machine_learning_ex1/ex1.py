from machine_learning_ex1.plot_data import plot_data
from machine_learning_ex1.warmup_exercise import warmup_exercise
from machine_learning_ex1.compute_cost import compute_cost
import numpy as np
import matplotlib

if __name__ == '__main__':
	"""
		Part 1: Basic Function
	"""
	print('Running warmup_exercise')
	print('5x5 Identity matrix:')
	mat = warmup_exercise()
	print(mat)

	"""
		Part 2: Plotting
	"""
	print('Plotting data...')
	data = np.loadtxt('ex1data1.txt', delimiter=',')
	m = np.shape(data)
	x = data[:,0]
	y = np.c_[data[:,1]]
	plot_data(x, y)

	"""
		Part 3: Cost And Gradient Descent
	"""
	x = np.c_[np.ones(data.shape[0]), data[:,0]]
	theta = np.zeros((2,1))
	iterations = 1500
	alpha = .01
	J = compute_cost(x, y, theta)
	print('cost: %0.4f ' % J)