__author__ = 'Hunter'
from machine_learning_ex2.sigmoid import sigmoid
import numpy as np

def predict(theta, x):
	m = len(x)
	p = np.zeros(m)

	h = sigmoid(x.dot(theta))
	for i in range(m):
		if h[i] >= 0.5:
			p[i] = 1
	return p