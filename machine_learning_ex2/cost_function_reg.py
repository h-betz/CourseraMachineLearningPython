__author__ = 'Hunter'
from machine_learning_ex2.sigmoid import sigmoid
import numpy as np

def cost_function_reg(theta, x, y, lamb):
	m = len(x)
	h = sigmoid(x.dot(theta))
	shape = (h.shape[0],1)
	j = (-1 / m) * np.sum((y * np.log(h).reshape(shape) + (1 - y) * np.log(1 - h).reshape(shape))) + np.sum((lamb / (2 * m)) * np.square(theta))
	return j