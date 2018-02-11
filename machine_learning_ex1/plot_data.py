import matplotlib.pyplot as plt

def plot_data(x, y):
	fig = plt.figure()
	ax = fig.add_subplot(1, 1, 1)
	ax.scatter(x, y)
	plt.show()
	# plt.plot(x,y, 'g^')
	# plt.ylabel('profit')
	# plt.xlabel('population')
	# plt.show()