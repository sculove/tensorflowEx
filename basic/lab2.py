import tensorflow as tf

# 1. create graph
# xTran = [1, 2, 3]
# yTran = [1, 2, 3]

x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

W = tf.Variable(tf.random_normal([1], name="weight"))
b = tf.Variable(tf.random_normal([1], name="bias"))

# H(x) = Wx + b
# hypothesis = xTran * W + b
hypothesis = x * W + b

# cost/loss function
# cost = tf.reduce_mean(tf.square(hypothesis - yTran))
cost = tf.reduce_mean(tf.square(hypothesis - y))

# optimizer
learning_rate = 0.01
optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate)
train = optimizer.minimize(cost)

# run learning
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(2001):
    # sess.run(train)
    # print(step, sess.run(cost), sess.run(W), sess.run(b))
    costVal, wVal, bVal, _ = sess.run([cost, W, b, train], feed_dict={x: [1, 2, 3, 4, 5], y: [2.1, 3.1, 4.1, 5.1, 6.1]})
    print(costVal, wVal, bVal)

# testing model
print("=== test model ===")
print(sess.run(hypothesis, feed_dict={x: [5]}))
print(sess.run(hypothesis, feed_dict={x: [2.5]}))
print(sess.run(hypothesis, feed_dict={x: [1.5, 3.5]}))