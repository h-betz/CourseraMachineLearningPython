__author__ = 'Hunter'
import numpy as np
import math

def compute_cost(x, y, theta):
	m = len(y)
	J = 0

	h = np.multiply(x, theta)
	sq_errors = math.pow((h - y), 2)
	J = (1 / (2*m)) * sum(sq_errors)
	return J