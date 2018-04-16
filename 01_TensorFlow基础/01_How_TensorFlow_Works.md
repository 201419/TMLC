# 使用TensorFlow的一般流程

1. 导入/生成样本数据集：毫无疑问，所有的机器学习算法都依赖样本数据集

2. 转换和归一化数据：一般来说，输入样本数据集不一定符合要求，使用前需要进行数据转换。TensorFlow有内建函数来归一化数据，如下

``` Python
data = tf.nn.batch_norm_with_global_normalization(...)
```

3. 划分样本数据集为训练样本集、测试样本集和验证样本集

4. 设置机器学习参数（超参数）：机器学习经常有一系列的常量参数，例如迭代次数、学习率等等

5. 初始化变量和占位符：在求解最优化过程中（最小化损失函数），TensorFlow通过占位符获取数据，并调整变量和权重。除了float32还有float16和float64（注意：使用的数据类型字节数越多，结果越精确，运行速度越慢）。

``` Python
a_var = tf.constant(42)
x_input = tf.placeholder(tf.float32, [None, input_size])
y_input = tf.placeholder(tf.float32, [None, num_classes])
```

6. 定义模型结构：TensorFlow通过操作、变量和占位符来构建计算图。

``` Python
y_pred = tf.add(tf.mul(x_input, weight_matrix), b_matrix)
```

7. 声明损失函数：定义完模型后，需要声明损失函数来评估输出结果。损失函数能够体现预测值和实际值的差距。例如

``` Python
loss = tf.reduce_mean(tf.square(y_actual - y_pred))
```

8. 初始化模型和训练模型

``` Python
with tf.Session(graph=graph) as session:
    ···
    session.run(...)
    ···
```

``` Python
session = tf.Session(graph=graph)
session.run(...)
```

9. 评估机器学习模型：模型训练完成之后，需要寻找某种标准来评估机器学习模型对新样本数据集的效果。通过评估，可以确定模型是过拟合还是欠拟合。

10. 调优超参数：通常情况下，需要基于模型效果来回调整参数，使用不同参数来训练模型，并用验证样本集来评估模型。

11. 发布/预测结果：所有机器学习模型一旦训练好，最后都用来预测新的、未知的数据。