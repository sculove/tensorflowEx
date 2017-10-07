import tensorflow as tf


W = tf.Variable(5.0, name="weight")
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

hypothesis = W * x
learning_rate = 0.01

# using tensorflow
cost = tf.reduce_mean(tf.square(hypothesis - y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate)

# optional tech
train = optimizer.minimize(cost)
# gvs = optimizer.compute_gradients(cost)
# train = optimizer.apply_gradients(gvs)

# manual minimize
# cost = tf.reduce_sum(tf.square(hypothesis - y))
# gradient = tf.reduce_mean((W * x - y) * x)
# descent = W - learning_rate * gradient
# train = W.assign(descent)

# run 
sess = tf.Session()
sess.run(tf.global_variables_initializer())
for step in range(100):
    wVal, _ = sess.run([W, train], feed_dict={x: [1, 2, 3], y: [1, 2, 3]})
    print(step, wVal)
