# Working with Bag of Words
#---------------------------------------
#
# 在此示例中，我们将下载并预处理 SMS Spam 文本数据。 
# 然后，我们将使用 独热编码（one-hot-encoding）和 逻辑回归 。
# 
# 我们将使用这些独热矢量（one-hot-vectors）进行逻辑回归
# 来预测文本是否为垃圾邮件（正常邮件标签：ham，垃圾邮件标签：spam）
# 
# 首先导入必要的库

import tensorflow as tf
import matplotlib.pyplot as plt
import os
import numpy as np
import string
from tensorflow.contrib import learn
from tensorflow.python.framework import ops
ops.reset_default_graph()

# Start a graph session
sess = tf.Session()

# 检查数据是否已经下载，否则下载数据
# （作者给出的文件已经不存在了，现在已经变成SMSSpamCollection，所以我进行了修改）
# 数据集链接：
# http://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip

# 数据集文件位置：./smsspamcollection/SMSSpamCollection
save_file_name = os.path.join('smsspamcollection','SMSSpamCollection')

# Create directory if it doesn't exist
if not os.path.exists('smsspamcollection'):
    print('Not Found ...')
    os.makedirs('smsspamcollection')

messages = [line.rstrip() for line in open(save_file_name, 'r', encoding='UTF-8')]  # 没有encoding的话报错UnicodeDecodeError
# print(len(messages))

# for message_no, message in enumerate(messages[:5]):
#     print(message_no, message)

# 这是一个 TSV 文件（用制表符 tab 分隔），
# 它的第一列是标记“正常信息”（ham）或“垃圾文件”（spam）的标签，第二列是信息本身
text_label = []
text_data = []
for message in messages:
    text = message.split('\t')
    text_label.append(text[0])
    text_data.append(text[1])
# print(text_label[:3])
# print(len(text_label))
# print(text_data[:3])
# print(len(text_data))

# 对数据进行预处理，例如删除标点和数字等等.
# 
# Relabel 'spam' as 1, 'ham' as 0
target = [1 if x=='spam' else 0 for x in text_label]
# 
# Normalize text
# 
# Lower case
texts = [x.lower() for x in text_data]
# 
# Remove punctuation and numbers
texts = [''.join(c for c in x if c not in string.punctuation) for x in texts]
texts = [''.join(c for c in x if c not in '0123456789') for x in texts]
# 
# Trim extra whitespace
texts = [' '.join(x.split()) for x in texts]

# print(target[:5])
# print(texts[:5])

# To determine a good sentence length to pad/crop at, 
# we plot a histogram of text lengths (in words).
# 为了确定填充/裁剪的良好句子长度，我们绘制文本长度的直方图（以单词表示）。

# Plot histogram of text lengths
# text_lengths = [len(x.split()) for x in texts]
# text_lengths = [x for x in text_lengths if x < 50]
# plt.hist(text_lengths, bins=25)
# plt.title('Histogram of # of Words in Texts')

# We crop/pad all texts to be 25 words long. 
# We also will filter out any words that do not appear at least 3 times.
# 我们将所有文本裁剪/填充为25个单词长度。我们还会过滤掉出现少于三次的单词。
# 
# Choose max text word length at 25
sentence_size = 25
min_word_freq = 3

# TensorFlow 有一个文本处理函数 VocabularyProcessor(). 我们使用这个函数来处理文本.
# 
# tensorflow.contrib.learn.preprocessing.VocabularyProcessor(
#                               max_document_length, 
#                               min_frequency = 0, 
#                               vocabulary = None, 
#                               tokenizer_fn = None)
# 
# 参数：
# max_document_length: 文档的最大长度。
#                      如果文本的长度大于最大长度，那么它会被剪切，反之则用0填充
# min_frequency: 词频的最小值，出现次数小于最小词频则不会被收录到词表中
# vocabulary: CategoricalVocabulary 对象
# tokenizer_fn：分词函数
# 
# Setup vocabulary processor
vocab_processor = learn.preprocessing.VocabularyProcessor(sentence_size, min_frequency=min_word_freq)

# Have to fit transform to get length of unique words.
trans = vocab_processor.transform(texts)
transformed_texts = np.array([x for x in trans])
embedding_size = len(np.unique(transformed_texts))

# 为了测试我们的 Logistic Model （预测邮件是spam还是ham），
# 将数据集分成训练集和测试集。
# 
# Split up data set into train/test
train_indices = np.random.choice(len(texts), round(len(texts)*0.8), replace=False)
test_indices = np.array(list(set(range(len(texts))) - set(train_indices)))
texts_train = [x for ix, x in enumerate(texts) if ix in train_indices]
texts_test = [x for ix, x in enumerate(texts) if ix in test_indices]
target_train = [x for ix, x in enumerate(target) if ix in train_indices]
target_test = [x for ix, x in enumerate(target) if ix in test_indices]

# ---

# Setup Index Matrix for one-hot-encoding
identity_mat = tf.diag(tf.ones(shape=[embedding_size]))

# Create variables for logistic regression
A = tf.Variable(tf.random_normal(shape=[embedding_size,1]))
b = tf.Variable(tf.random_normal(shape=[1,1]))

# Initialize placeholders
x_data = tf.placeholder(shape=[sentence_size], dtype=tf.int32)
y_target = tf.placeholder(shape=[1, 1], dtype=tf.float32)

# Text-Vocab Embedding
x_embed = tf.nn.embedding_lookup(identity_mat, x_data)
x_col_sums = tf.reduce_sum(x_embed, 0)

# Declare model operations
x_col_sums_2D = tf.expand_dims(x_col_sums, 0)
model_output = tf.add(tf.matmul(x_col_sums_2D, A), b)

# Declare loss function (Cross Entropy loss)
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=model_output, labels=y_target))

# Prediction operation
prediction = tf.sigmoid(model_output)

# Declare optimizer
train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss)

# Intitialize Variables
sess.run(tf.global_variables_initializer())

# Start Logistic Regression
print('Starting Training Over {} Sentences.'.format(len(texts_train)))
loss_vec = []
train_acc_all = []
train_acc_avg = []

for ix, t in enumerate(vocab_processor.fit_transform(texts_train)):
    y_data = [[target_train[ix]]]
    
    sess.run(train_step, feed_dict={x_data: t, y_target: y_data})
    temp_loss = sess.run(loss, feed_dict={x_data: t, y_target: y_data})
    loss_vec.append(temp_loss)
    
    if (ix+1)%10 == 0:  # 每隔10输出一次loss值
        print('Training Observation #' + str(ix+1) + ': Loss = ' + str(temp_loss))
        
    # Keep trailing average of past 50 observations accuracy
    # Get prediction of single observation
    [[temp_pred]] = sess.run(prediction, feed_dict={x_data:t, y_target:y_data})
    # Get True/False if prediction is accurate
    train_acc_temp = target_train[ix]==np.round(temp_pred)
    train_acc_all.append(train_acc_temp)
    if len(train_acc_all) >= 50:
        train_acc_avg.append(np.mean(train_acc_all[-50:]))

# 训练好 logistic 模型以后，就可以在测试集上运行模型得到准确率。
# 
# Get test set accuracy
print('Getting Test Set Accuracy For {} Sentences.'.format(len(texts_test)))
test_acc_all = []

for ix, t in enumerate(vocab_processor.fit_transform(texts_test)):
    y_data = [[target_test[ix]]]
    
    if (ix+1)%50 == 0:  # 每隔50输出一次
        print('Test Observation #' + str(ix+1))
    
    # Get prediction of single observation
    [[temp_pred]] = sess.run(prediction, feed_dict={x_data:t, y_target:y_data})
    # Get True/False if prediction is accurate
    test_acc_temp = target_test[ix]==np.round(temp_pred)
    test_acc_all.append(test_acc_temp)

print('\nOverall Test Accuracy: {}'.format(np.mean(test_acc_all)))

# Plot training accuracy over time
plt.plot(range(len(train_acc_avg)), train_acc_avg, 'k-', label='Train Accuracy')
plt.title('Avg Training Acc Over Past 50 Iterations')
plt.xlabel('Iterations')
plt.ylabel('Training Accuracy')
plt.show()