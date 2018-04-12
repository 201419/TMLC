# 使用TensorFlow的一般流程

1. 导入/生成样本数据集

2. 转换和归一化数据

''' Python
data = tf.nn.batch_norm_with_global_normalization(...)
'''

3. 划分样本数据集为训练样本集、测试样本集和验证样本集

4. 设置机器学习参数（超参数）

5. 初始化变量和占位符

''' Python
a_var = tf.constant(42)
x_input = tf.placeholder(tf.float32, [None, input_size])
y_input = tf.placeholder(tf.float32, [None, num_classes])
'''

6. 定义模型结构

7. 声明损失函数

8. 初始化模型和训练模型

9. 评估机器学习模型

10. 调优超参数