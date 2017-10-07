import tensorflow as tf

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
addNode = a + b
sess = tf.Session()

print("===result===")
print(sess.run(addNode, feed_dict={a: 3, b: 4}))
print(sess.run(addNode, feed_dict={a: [1,3], b: [2, 4]}))