__author__ = 'Hunter'
import numpy as np

def compute_cost_multi(x, y, theta):
	m = y.size
	h = x.dot(theta)
	sq_errors = np.square(h - y)
	J = (1 / (2*m)) * np.sum(sq_errors)
	return J