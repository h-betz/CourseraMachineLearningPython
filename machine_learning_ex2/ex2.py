from machine_learning_ex2.plot_data import plot_data
from machine_learning_ex2.cost_function import cost_function
from machine_learning_ex2.gradient_function import gradient_function
from machine_learning_ex2.sigmoid import sigmoid
from machine_learning_ex2.predict import predict
from scipy.optimize import minimize
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
	plot_data(x, y)

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
	initial_theta = np.zeros(n+1)

	cost = cost_function(initial_theta, x, y)
	grad = gradient_function(initial_theta, x, y)

	print('Cost at initial theta (zeros): %s' % cost)
	print('Expected cost (approx): 0.693')
	print('Gradient at initial theta (zeros): ')
	print('%s' % grad)
	print('Expected gradients (approx):\n -0.1000\n -12.0092\n -11.2628\n')

	# test_theta = np.c_[[-24, 0.2, 0.2]]
	#
	# cost = cost_function(test_theta, x, y)
	# grad = gradient_function(test_theta, x, y)
	# print('Cost at test theta: %s' % cost)
	# print('Expected cost (approx): 0.218')
	# print('Gradient at test theta:')
	# print('%s' % grad)
	# print('Expected gradients (approx):\n 0.043\n 2.566\n 2.647')


	"""
		Part 3: Optimizing using fminunc

		In this exercise, you will use a built-in function (fminunc) to find the
		optimal parameters theta.
	"""
	res = minimize(cost_function, initial_theta, args=(x,y), method=None, jac=gradient_function, options={'maxiter':400})
	print('Cost at theta found by fminunc: %s' % res.fun)
	print('Expected cost (approx): 0.203')
	print('theta: %s' % res.x)
	print('Expected theta (approx):')
	print(' -25.161\n 0.206\n 0.201')

	"""
		Part 4: PRedict and Accuracies

		After learning the parameters, you'll like to use it to predict the outcomes
		on unseen data. In this part, you will use the logistic regression model
		to predict the probability that a student with score 45 on exam 1 and
		score 85 on exam 2 will be admitted.

		Furthermore, you will compute the training and test set accuracies of our model.
		Your task is to complete the code in predict.py
		Predict probability for a student with score 45 on exam 1 and score 85 on exam 2
	"""
	theta = res.x
	prob = sigmoid(np.array([1, 45, 85]).dot(theta))
	print('For a student with scores 45 and 85, we predict an admission probability of %s' % prob)
	print('Expected value: 0.775 +/- 0.002')

	p = predict(theta, x)

	acc = 100*sum(p == y.ravel()) / p.size
	print('Train Accuracy: %s' % acc)
	print('Expected accuracy (approx): 89.0')