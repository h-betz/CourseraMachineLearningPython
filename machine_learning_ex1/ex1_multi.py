from machine_learning_ex1.feature_normalize import feature_normalize
from machine_learning_ex1.gradient_descent_multi import gradient_descent_multi
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

if __name__ == '__main__':
	"""
		Part 1: Feature Normalization
	"""
	data = np.loadtxt('ex1data2.txt', delimiter=',')
	x = data[:,:2]
	y = data[:,2]
	m = len(y)

	print('First 10 examples from the dataset:')
	# print('x = [%s %s], y = %s \n' % (x[:10,:], y[:10,:]))
	print('Normalizing Features ...')

	x, mu, sigma, = feature_normalize(x)

	# Add intercept term to X
	x = np.concatenate((np.ones((m, 1)), x), axis=1)

	"""
		Part 2: Gradient Descent 
		
		Instructions: 
		We have provided you with the following starter	code that runs gradient descent with a particular
		learning rate (alpha). 
		
		Your task is to first make sure that your functions - computeCost and
		gradientDescent already work with this starter code and support multiple variables.
		
		After that, try running gradient descent with different values of alpha and see which one gives
		you the best result.
		
		Finally, you should complete the code at the end to predict the price of a 1650 sq-ft, 3 br house.
		
		Hint: At prediction, make sure you do the same feature normalization.
	"""
	print('Running gradient descent ...')

	# Choose some alpha value
	alpha = 0.01
	num_iters = 400

	# Init Theta and Run Gradient Descent
	theta = np.zeros((3, 1))
	theta, j_history = gradient_descent_multi(x, y, theta, alpha, num_iters)
	plt.plot(j_history, '-b')
	plt.xlabel('Number of iterations')
	plt.ylabel('Cost J')
	plt.show()

	print('Theta computed from gradient descent: ')
	print('%s' % theta)

	"""
		Estimate the price of a 1650 sq-ft, 3 br house
	"""
	d = np.array([1650, 3])
	d = (d - mu) / sigma
	d = np.append(1, d)
	price = d * theta