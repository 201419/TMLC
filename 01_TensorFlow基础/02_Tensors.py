# 张量（Tensors）
# 这里将介绍在TensorFlow中创建张量的一些方法。

import tensorflow as tf

sess = tf.Session()

# 1. 固定张量

# 创建指定维度的零张量，tf.zeros([row_dim, col_dim])
zero_tsr = tf.zeros([3,5])
print(zero_tsr)
print(sess.run(zero_tsr))

# 创建指定维度的单位张量，tf.ones([row_dim, col_dim])
ones_tsr = tf.ones([3,5])
print(ones_tsr)
print(sess.run(ones_tsr))

# 创建指定维度的常数填充的张量，tf.fill([row_dim, col_dim], constant)
filled_tsr = tf.fill([3,5], 42)
print(filled_tsr)
print(sess.run(filled_tsr))

# 用已知常数张量创建一个张量，tf.constant([constant, ... , constant])
constant_tsr = tf.constant([1, 2, 3])
print(constant_tsr)
print(sess.run(constant_tsr))

# 2. 相似形状的张量

zeros_similar = tf.zeros_like(constant_tsr)
print(zeros_similar)
print(sess.run(zeros_similar))

ones_similar = tf.ones_like(constant_tsr)
print(ones_similar)
print(sess.run(ones_similar))

# 3. 序列张量

# Linspace in TensorFlow
linear_tsr = tf.linspace(start=0.0, stop=1.0, num=3) # 生成 [0.0, 0.5, 1.0] ，包含stop
print(linear_tsr)
print(sess.run(linear_tsr))
print('-----------------------------')
# Range in TensorFlow
sequence_var = tf.Variable(tf.range(start=6.0, limit=15.0, delta=3.24)) # 生成 [6, 9, 12] ，不包含limit
sess.run(sequence_var.initializer)
print(sequence_var)
print(sess.run(sequence_var))
# sequence_var在这里是一个变量， 使用tf.Variable来将张量封装成一个变量
# 声明变量之后需要初始化才能使用
# 也可以使用下边的方式来声明
# initialize_op = tf.global_variables_initializer()
# sess.run(initialize_op)

# 4. 随机张量

#  tf.random_normal生成正态分布的随机数
rnorm_tsr = tf.random_normal([3, 5], mean=0.0, stddev=1.0)
print(rnorm_tsr)
print(sess.run(rnorm_tsr))
print('-----------------------------')
#  tf.random_uniform生成均匀分布的随机数，从minval（包含minval）到maxval（不包含maxval）
runif_tsr = tf.random_uniform([3, 5], minval=0, maxval=4)
print(runif_tsr)
print(sess.run(runif_tsr))

#  tf.truncated_normal生成带有指定边界的正态分布的随机数
# 其正态分布的随机数位于指定均值（期望）到两个标准差之间的区间
rTruncNorm_tsr = tf.truncated_normal([3, 5], mean=0.0, stddev=1.0)
print(rTruncNorm_tsr)
print(sess.run(rTruncNorm_tsr))

# 张量/数组的随机化
input_tensor = tf.constant([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
shuffled_output = tf.random_shuffle(input_tensor)
print(shuffled_output)
print(sess.run(shuffled_output))
print('-----------------------------')
cropped_output = tf.random_crop(input_tensor, [1, 3])
print(cropped_output)
print(sess.run(cropped_output))
# tf.random_crop可以实现对张量指定大小的随机裁剪