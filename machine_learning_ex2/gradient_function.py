__author__ = 'Hunter'
from machine_learning_ex2.sigmoid import sigmoid
import numpy as np

def gradient_function(theta, x, y):
	m = len(x)
	h = sigmoid(x.dot(theta.reshape(-1,1)))
	grad = (1 / m) * x.T.dot(h - y)
	return grad.flatten()
