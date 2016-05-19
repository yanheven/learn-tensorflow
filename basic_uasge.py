import tensorflow as tf


# matrix1 = tf.constant([[3, 3]])
# matrix2 = tf.constant([[2], [2]])

matrix1 = tf.constant(9**9, shape=[10, 1000000])
matrix2 = tf.constant(9*9, shape=[1000000, 10])

product = tf.matmul(matrix1, matrix2)
# sess = tf.Session()
# result = sess.run(product)
# print(result)
# sess.close()

with tf.Session() as sess:
    cpu_str = '/cpu:'
    for i in range(3):
        with tf.device(cpu_str + str(i)):
            matrix1 = tf.constant(9, shape=[100, 10000])
            matrix2 = tf.constant(9, shape=[10000, 100])

            product = tf.matmul(matrix1, matrix2)
            print('Runing on Processor: {0}'.format(i))
            result = sess.run(product)

print(result)