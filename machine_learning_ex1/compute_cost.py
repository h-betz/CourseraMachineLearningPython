__author__ = 'Hunter'
import numpy as np
import math

def compute_cost(x, y, theta):
	m = y.size
	h = x.dot(theta)
	sq_errors = np.square(h - y)
	J = (1 / (2*m)) * np.sum(sq_errors)
	return J