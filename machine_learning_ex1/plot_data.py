import matplotlib.pyplot as plt

def plot_data(x, y):
	plt.plot(x,y, 'g^')
	plt.ylabel('profit')
	plt.xlabel('population')
	plt.show()