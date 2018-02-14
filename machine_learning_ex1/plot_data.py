import matplotlib.pyplot as plt

def plot_data(x, y):
	fig = plt.figure()
	ax = fig.add_subplot(1, 1, 1)
	ax.scatter(x, y)
	plt.show()