# Working with Bag of Words

为了说明如何使用 "bag of words" 处理文本数据，我们将使用来自UCI的 "[SMS Spam Collection Data Set](http://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection)"。 

The SMS Spam Collection is a public set of SMS labeled messages that have been collected for mobile phone spam research.

下载此数据集以备接下来使用，然后我们会使用 "bag of words" 方法来预测文本是否为垃圾邮件。 

The model that will operate on the bag of words will be a logistic model with no hidden layer.

我们将使用批量大小为1的随机训练，最后来计算测试集的准确度。

---

**以下内容来自维基百科**

[词袋模型](https://zh.wikipedia.org/wiki/%E8%AF%8D%E8%A2%8B%E6%A8%A1%E5%9E%8B)

词袋模型（英语：Bag-of-words model）是个在自然语言处理和信息检索(IR)下被简化的表达模型。此模型下，一段文本（比如一个句子或是一个文档）可以用一个装着这些词的袋子来表示，这种表示方式不考虑文法以及词的顺序。最近词袋模型也被应用在计算机视觉领域。

词袋模型被广泛应用在文件分类，词出现的频率可以用来当作训练分类器的特征。

关于"词袋"这个用字的由来可追溯到泽里格·哈里斯于1954年在Distributional Structure的文章。

范例：

下列文件可用词袋表示:

以下是两个简单的文件:

 - (1) John likes to watch movies. Mary likes movies too.
 - (2) John also likes to watch football games.

基于以上两个文件，可以建构出下列清单:

[
    "John",
    "likes",
    "to",
    "watch",
    "movies",
    "also",
    "football",
    "games",
    "Mary",
    "too"
]

此处有10个不同的词，使用清单的索引表示长度为10的向量:

 - (1) [1, 2, 1, 1, 2, 0, 0, 0, 1, 1] 
 - (2) [1, 1, 1, 1, 0, 1, 1, 1, 0, 0] 

每个向量的索引内容对应到清单中词出现的次数。

举例来说，第一个向量(文件一)前两个内容索引是1和2，第一个索引内容是"John"对应到清单第一个词并且该值设定为1，因为"John"出现一次。

此向量表示法不会保存原始句子中词的顺序。该表示法有许多成功的应用，像是邮件过滤。

在上述的范例，文件向量包含term频率 。 在IR和文字分类常用不同方法来量化term权重。 常见方法为tf-idf。