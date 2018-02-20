__author__ = 'Hunter'
import numpy as np

def normal_eqn(x, y):
	theta = np.linalg.inv(x.T.dot(x)).dot(x.T).dot(y)
	return theta
