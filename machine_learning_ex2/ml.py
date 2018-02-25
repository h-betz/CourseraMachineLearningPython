__author__ = 'Hunter'
from pandas import Series

def map_feature(x, degree=6):
	"""
	Feature mapping function to polynomial features

	MAPFEATURE(X, degree) maps the two input features
	to quadratic features used in the regularization exercise.

	Returns a new feature array with more features, comprising of
	X1, X2, X1.^2, X2.^2, X1*X2, X1*X2.^2, etc..
	:param x:
	:param degree:
	:return:
	"""
	quads = Series([x.iloc[0]**(i-j) * x.iloc[1]**j for i in range(1, degree+1) for j in range(i+1)])
	return Series([1]).append([x, quads])
