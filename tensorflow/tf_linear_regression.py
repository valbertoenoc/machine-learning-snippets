import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('tkAgg')
from matplotlib import pyplot as plt

# data
#trainX = np.linspace(-1,1,101)
#trainY = 3 * trainX + np.random.randn(*trainX.shape) * 0.33
data = np.loadtxt('../data/aerogerador.dat')
trainX = data[:,0]
trainY = data[:,1]
n_samples = trainX.shape[0]

# tensor initialization
X = tf.placeholder('float')
Y = tf.placeholder('float')

# model
w = tf.Variable(np.random.randn(), name='weights')
b = tf.Variable(np.random.randn(), name='bias')
y_model = tf.add(tf.multiply(X, w), b)

# optimizing cost function (GD)
# cost = tf.pow(Y-y_model, 2)
cost = tf.reduce_sum(tf.pow(y_model-Y, 2))/(2*n_samples)

learning_rate = 0.01
# optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

# initialize tensor variables
init = tf.global_variables_initializer()

# running session
epochs = 100
with tf.Session() as sess:
	sess.run(init)
	for i in range(epochs):
		for (x,y) in zip(trainX, trainY):
			sess.run(optimizer, feed_dict={X: x, Y: y})

		# print log per epochs
		if (i + 1) % 10 == 0:
			c = sess.run(cost, feed_dict={X: trainX, Y:trainY})
			print("Epoch:", '%4d' % (i+1), 'cost=', '{:.9}'.format(c), \
			'W=', sess.run(w), 'b=', sess.run(b))

	print('Optimization finised.')
	training_cost = sess.run(cost, feed_dict={X: trainX, Y: trainY})
	print('Training cost=', training_cost, 'W=', sess.run(w), 'b=', sess.run(b))

	# display graphic
	plt.plot(trainX, trainY, 'ro', label='Original Data')
	plt.plot(trainX, sess.run(w)*trainX + sess.run(b), label='Fitted function')
	plt.legend()
	plt.show()

