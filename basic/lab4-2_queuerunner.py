import tensorflow as tf

queue = tf.train.string_input_producer(["basic/data/test-score.csv"], shuffle=False)

reader = tf.TextLineReader()
key, value = reader.read(queue)

raw = tf.decode_csv(value, record_defaults= [
  [0.],
  [0.],
  [0.],
  [0.]
  ])

# collect batches of csv in
train_X_batch, train_Y_batch = tf.train.batch([raw[0:-1], raw[-1:]], batch_size = 10)
# X_data = raw[:, 0:-1]
# Y_data = raw[:, [-1]]

X = tf.placeholder(tf.float32, shape=[None, 3])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([3, 1]), name = "weight")
b = tf.Variable(tf.random_normal([1]), name = "bias")
hypothesis = tf.matmul(X, W) + b

# cost function
cost = tf.reduce_mean(tf.square(hypothesis - Y))
# optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

# Launch
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# start queue
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess = sess, coord = coord)

for step in range(2001):
  X_data, Y_data = sess.run([train_X_batch, train_Y_batch])
  cost_val, hy_val, _ = sess.run([cost, hypothesis, train], feed_dict={X:X_data, Y:Y_data})

  if step % 10 == 0:
    print(step, "cost: " , cost_val, "\nhypothesis : ", hy_val)

coord.request_stop()
coord.join(threads)

# testing
print("==== testing ====")
print(sess.run(hypothesis, feed_dict={X: [[100, 70, 101]]}))
print(sess.run(hypothesis, feed_dict={X: [[60, 70, 110], [90, 100, 80]]}))