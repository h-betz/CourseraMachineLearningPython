from machine_learning_ex1.compute_cost import compute_cost
import numpy as np

def gradient_descent(x, y, theta, alpha, iterations):
	m = y.size
	j_history = np.zeros((iterations,1))

	for j in range(1,iterations):
		h = x.dot(theta)
		delta = (1 / m) * (x.T.dot(h-y))
		theta = theta - alpha * delta
		j_history[j] = compute_cost(x, y, theta)
	return theta, j_history