from machine_learning_ex1.plot_data import plot_data
from machine_learning_ex1.warmup_exercise import warmup_exercise
from machine_learning_ex1.compute_cost import compute_cost
from machine_learning_ex1.gradient_descent import gradient_descent
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
	print('With theta = [0 ; 0]\nCost computed = %s\n' % J)
	print('Expected cost value (approx) 32.07\n')
	J = compute_cost(x, y, [[-1],[2]])
	print('With theta = [-1 ; 2]\nCost computed = %s\n' % J)
	print('Expected cost value (approx) 54.24')
	print('Running Gradient Descent ...')
	theta, j_history = gradient_descent(x, y, theta, alpha, iterations)
	print('Theta found by gradient descent:')
	print('%s' % theta)
	print('Expected theta values (approx):')
	print('-3.6303\n  1.1664')
	plot_data(x[:,1], x.dot(theta))
	predict_1 = np.matrix('1 3.5') * theta
	print('For population = 35,000, we predict a profit of %s' % (predict_1*10000))
	predict_2 = np.matrix('1 7') * theta

	"""
		Part 4: Visualizing J(theta_0, theta_1)
	"""
	print('Visualizing J(theta_0, theta_1) ...')

	# Grid over which we will calculate J
	theta0_vals = np.linspace(-10, 10, 100)
	theta1_vals = np.linspace(-1, 4, 100)

	# initialize J_vals to a matrix of 0's
	j_vals = np.zeros((len(theta0_vals), len(theta1_vals)))

	# Fill out J_vals
	for i in range(len(theta0_vals)):
		for j in range(len(theta1_vals)):
			t = np.matrix('%s;%s' % (theta0_vals[i], theta1_vals[j]))
			j_vals[i,j] = compute_cost(x, y, t)