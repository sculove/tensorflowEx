import tensorflow as tf
import numpy as np

raw = np.loadtxt("basic/data/test-score.csv", delimiter = ",", dtype = np.float32)
X_data = raw[:, 0:-1]
Y_data = raw[:, [-1]]

xLen = len(X_data[0])
yLen = len(Y_data[0])

X = tf.placeholder(tf.float32, shape=[None, xLen])
Y = tf.placeholder(tf.float32, shape=[None, yLen])

W = tf.Variable(tf.random_normal([xLen, yLen]), name = "weight")
b = tf.Variable(tf.random_normal([yLen]), name = "bias")
hypothesis = tf.matmul(X, W) + b

# cost function
cost = tf.reduce_mean(tf.square(hypothesis - Y))
# optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)


# Launch
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(2001):
  cost_val, hy_val, _ = sess.run([cost, hypothesis, train], feed_dict={X:X_data, Y:Y_data})

  if step % 10 == 0:
    print(step, "cost: " , cost_val, "\nhypothesis : ", hy_val)

# testing
print("==== testing ====")
print(sess.run(hypothesis, feed_dict={X: [[100, 70, 101]]}))
print(sess.run(hypothesis, feed_dict={X: [[60, 70, 110], [90, 100, 80]]}))