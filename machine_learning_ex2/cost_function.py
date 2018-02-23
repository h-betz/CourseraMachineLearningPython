from machine_learning_ex2.sigmoid import sigmoid
import numpy as np

"""
	Compute cost and gradient for logistic regression
	J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
	parameter for logistic regression and the gradient of the cost
	w.r.t. to the parameters.
"""

def cost_function(theta, x, y):
	# Compute the cost of a particular choice of theta.You should set J to the cost.
	# Compute the partial derivatives and set grad to the partial derivatives of the
	# cost w.r.t. each parameter in theta

	m = len(y)

	h = sigmoid(x.dot(theta))
	j = (-1 / m) * (np.multiply(y, np.log(h)) + (1 - y) * np.log(1 - h))

	grad = (1 / m) * x.T.dot(h - y)

	return j, grad