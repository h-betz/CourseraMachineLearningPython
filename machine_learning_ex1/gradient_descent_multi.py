from machine_learning_ex1.compute_cost_multi import compute_cost_multi
import numpy as np

def gradient_descent_multi(x, y, theta, alpha, num_iters):
	# TODO complete function
	m = len(y)
	j_history = np.zeros((num_iters, 1))

	for j in range(1, num_iters):
		h = x.dot(theta)
		delta = (1 / m) * (x.T.dot(h-y))
		theta = theta - (alpha * delta)
		j_history[j] = compute_cost_multi(x, y, theta)

	return theta, j_history