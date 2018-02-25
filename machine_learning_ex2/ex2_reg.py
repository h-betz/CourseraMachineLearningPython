__author__ = 'Hunter'
from machine_learning_ex2.plot_data import plot_data
from machine_learning_ex2.ml import map_feature
from machine_learning_ex2.cost_function_reg import cost_function_reg
from machine_learning_ex2.gradient_function_reg import gradient_function_reg
import pandas as pd
import numpy as np

if __name__ == "__main__":
	data = pd.read_csv('ex2data2.txt', header=None, names=[1,2,3])
	x = data[[1,2]]
	y = data[[3]]
	plot_data(x.values, y.values)

	"""
		Part 1: Regularized Logistic Regression

		In this part, you are given a dataset with data points that are not
		linearly separable. However, you would still like to use logistic
		regression to classify the data points.

		To do so, you introduce more features to use -- in particular, you add
		polynomial features to our data matrix (similar to polynomial regression).

	"""

	x = x.apply(map_feature, axis=1)
	initial_theta = np.zeros(x.shape[1])

	lamb = 0.0

	cost = cost_function_reg(initial_theta, x.values, y.values, lamb)
	grad = gradient_function_reg(initial_theta, x.values, y.values, lamb)

	print('Cost at initial theta (zeros): %s' % cost)
	print('Expected cost (approx): 0.693')
	print('Gradient at initial theta (zeros) - first five values only')
	print('%s' % grad[:5])
	print('Expected gradients (approx) - first five values only:')
	print('0.0085\n 0.0188\n 0.0001\n 0.0503\n 0.0115')

	"""
		Part 2: Regularization and Accuracies

		Optional Exercise:
		In this part, you will get to try different values of lambda and
		see how regularization affects the decision coundart

		Try the following values of lambda (0, 1, 10, 100).
		How does the decision boundary change when you vary lambda? How does
		the training set accuracy vary?
	"""

	# Initialize fitting parameters
	initial_theta = np.zeros((x.shape[1], 1))
	lamb = 1

