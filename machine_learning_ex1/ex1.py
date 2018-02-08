from machine_learning_ex1.plot_data import plot_data
from machine_learning_ex1.warmup_exercise import warmup_exercise

if __name__ == '__main__':
	print('Running warmup_exercise')
	print('5x5 Identity matrix:')
	mat = warmup_exercise()
	print(mat)
	print('Plotting data...')
	f = open('ex1data1.txt', 'rU')
	x, y = [], []
	for r in f.readlines():
		x.append(r.split(',')[0])
		y.append(r.split(','))[1]
	f.close()
	plot_data(x, y)