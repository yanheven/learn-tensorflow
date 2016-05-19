import tensorflow as tf
import tensorflow.python.platform
hello = tf.constant('Hello, tensorflow!!')
sess = tf.Session()
print(sess.run(hello))

a = tf.constant(11)
b = tf.constant(22)
print(sess.run(a+b))