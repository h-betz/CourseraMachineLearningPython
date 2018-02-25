__author__ = 'Hunter'
from machine_learning_ex2.sigmoid import sigmoid
import numpy as np

def gradient_function_reg(theta, x, y, lamb):
	m = len(x)
	n = len(theta)
	grad = np.zeros(n)
	h = sigmoid(x.dot(theta))
	grad[0] = (1 / m) * (np.sum(x.T[0,:].dot(h) - x.T[0,:].dot(y)))
	for j in range(1,n):
		grad[j] = (1 / m) * (np.sum(x.T[j,:].dot(h) - x.T[j,:].dot(y)) + lamb * theta[j])
	return grad