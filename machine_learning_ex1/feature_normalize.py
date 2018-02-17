import numpy as np

def feature_normalize(x):
	"""
	   returns a normalized version of X where
	   the mean value of each feature is 0 and the standard deviation
	   is 1. This is often a good preprocessing step to do when
	   working with learning algorithms.
	"""

	# ====================== YOUR CODE HERE ======================
	# Instructions: First, for each feature dimension, compute the mean
	#               of the feature and subtract it from the dataset,
	#               storing the mean value in mu. Next, compute the
	#               standard deviation of each feature and divide
	#               each feature by it's standard deviation, storing
	#               the standard deviation in sigma.
	#
	#               Note that X is a matrix where each column is a
	#               feature and each row is an example. You need
	#               to perform the normalization separately for
	#               each feature.
	#
	# Hint: You might find the 'mean' and 'std' functions useful.
	#
	x_norm, mu, sigma = x, np.zeros((1,2)), np.zeros((1,2))
	mu = x.mean(axis=0)
	sigma = x.std(axis=0)
	for i in range(len(x)):
		x_norm[i] = (x[i] - mu) / sigma

	# ============================================================
	return x_norm, mu, sigma