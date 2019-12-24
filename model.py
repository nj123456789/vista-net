import tensorflow as tf
import tensorflow.contrib.rnn as rnn

from data_utils import batch_review_normalize, batch_image_normalize
from layers import bidirectional_rnn, text_attention, visual_aspect_attention
from model_utils import get_shape, load_glove

from data_preprocess import VOCAB_SIZE

#创建vistanet类，用以创建实例对象
class VistaNet: 

  
  '''
  由于类可以起到模板的作用，因此，可以在创建实例的时候，把一些我们认为必须绑定的属性强制填写进去。
  通过定义一个特殊的__init__方法，在创建实例的时候，就把hidden_dim，att_dim等属性绑上去：
  注意到__init__方法的第一个参数永远是self，表示创建的实例本身，因此，在__init__方法内部，
  就可以把各种属性绑定到self，因为self就指向创建的实例本身
  '''
  def __init__(self, hidden_dim, att_dim, emb_size, num_images, num_classes):#括号内参数是捆绑在self上的
    self.hidden_dim = hidden_dim
    self.att_dim = att_dim
    self.emb_size = emb_size
    self.num_classes = num_classes
    self.num_images = num_images
    
    
    
#除了定义中的参数，接下来还定义了如下参数（实例的属性）
     '''
     当神经网络的结构更加复杂、参数更多时，就需要一个更好的方式来传递和管理神经网络中的参数了。TensorFlow提
     供了通过变量名称（name）来创建或者获取一个变量的机制。通过这个机制，在不同的函数中可以直接通过变量的名字
     来使用变量，而不需要将变量通过参数的形式到处传递。tensorflow中定义变量的方式与我们常规理解不一样，而是
     通过tf.Variable方法定义变量，参数0表示变量初始值，trainable=False表示
     后期不用优化变量，否则置true，会把它加入到GraphKeys.TRAINABLE_VARIABLES，能对它使用Optimizer。
     '''
    self.global_step = tf.Variable(0, name='global_step', trainable=False)
    
    
    
    
      '''
      Tensorflow的设计理念称之为计算流图，在编写程序时，首先构筑整个系统的graph，代码并不会直接生效，这一点和python的其他数值计算库
     （如Numpy等）不同，graph为静态的，类似于docker中的镜像。然后，在实际的运行时，启动一个session，程序才会真正的运行。这样做的好
      处就是：避免反复地切换底层程序实际运行的上下文，tensorflow帮你优化整个系统的代码。
      placeholder()函数是在神经网络构建graph的时候在模型中的占位，此时并没有把要输入的数据传入模型，它只会分配必要的内存。等建立session，
      在会话中，运行模型的时候通过feed_dict()函数向占位符喂入数据。
      dtype：数据类型。常用的是tf.float32,tf.float64等数值类型
      shape：数据形状。默认是None，就是一维值，也可以是多维（比如[2,3], [None, 3]表示列是3，行不定）
      name：名称
      '''
    self.dropout_keep_prob = tf.placeholder(dtype=tf.float32, name='dropout_keep_prob')

    self.documents = tf.placeholder(shape=(None, None, None), dtype=tf.int32, name='reviews')
    self.document_lengths = tf.placeholder(shape=(None,), dtype=tf.int32, name='review_lengths')
    self.sentence_lengths = tf.placeholder(shape=(None, None), dtype=tf.int32, name='sentence_lengths')

    self.max_num_words = tf.placeholder(dtype=tf.int32, name='max_num_words')
    self.max_num_sents = tf.placeholder(dtype=tf.int32, name='max_num_sents')

    self.images = tf.placeholder(shape=(None, None, 4096), dtype=tf.float32, name='images')
    self.labels = tf.placeholder(shape=(None), dtype=tf.int32, name='labels')
    
    
    '''
    上下文管理器是一个包装任意代码块的对象。上下文管理器保证进入上下文管理器时，每次代码执行的一致性；
    当退出上下文管理器时，相关的资源会被正确的回收。tf.variable_scope本质上就是一个上下文管理器
    tf.variable_scope(): 可以让变量有相同的命名，包括tf.get_variable得到的变量，还有tf.Variable变量
    它返回的是一个用于定义创建variable(层)的op的上下文管理器。
    可变范围允许创建新的variable并分享已创建的variable，同时提供检查，不会意外创建或共享。
    '''
    with tf.variable_scope('VistaNet'):
      self._init_embedding()
      self._init_word_encoder()
      self._init_sent_encoder()
      self._init_classifier()
      
      
  #定义实例同上，但是没有捆绑属性
  def _init_embedding(self):
    with tf.variable_scope('embedding'):
      
  '''   
  get_variable获取一个已经存在的变量或者创建一个变量
  name：新变量或现有变量的名称。
  shape：新变量或现有变量的形状。
  dtype：新变量或现有变量的类型（默认为DT_FLOAT）。
  ininializer：如果创建了则用它来初始化变量。
  '''
      self.embedding_matrix = tf.get_variable(
        name='embedding_matrix',
        shape=[VOCAB_SIZE, self.emb_size],
        initializer=tf.constant_initializer(load_glove(VOCAB_SIZE, self.emb_size)),
        dtype=tf.float32
      )
    
    
  '''  
  张量是一个非常复杂的概念。为了宝贝理解，粗浅的、不准确的，但是非常简单形象的理解张量就是张量是有大小和多个方向的量。
  tf.nn.embedding_lookup()函数的用法主要是选取一个张量里面索引对应的元素，
  tf.nn.embedding_lookup(tensor,id)：即tensor就是输入的张量，id 就是张量对应的索引
  '''
      self.embedded_inputs = tf.nn.embedding_lookup(self.embedding_matrix, self.documents)

  def _init_word_encoder(self):
    
    
    '''
    这两段代码帮你理解：
    一：
    with tf.name_scope('conv1') as scope:
       weights1 = tf.Variable([1.0, 2.0], name='weights')
       bias1 = tf.Variable([0.3], name='bias')
    #下面是在另外一个命名空间来定义变量的
    with tf.name_scope('conv2') as scope:
       weights2 = tf.Variable([4.0, 2.0], name='weights')
       bias2 = tf.Variable([0.33], name='bias')
    所以，实际上weights1 和 weights2 这两个引用名指向了不同的空间，不会冲突
    
    二：
    with tf.variable_scope('v_scope') as scope1:
       Weights1 = tf.get_variable('Weights', shape=[2,3])
       bias1 = tf.get_variable('bias', shape=[3])
    # 下面来共享上面已经定义好的变量
    # note: 在下面的 scope 中的变量必须已经定义过了，才能设置 reuse=True，否则会报错
    with tf.variable_scope('v_scope', reuse=True) as scope2:
      Weights2 = tf.get_variable('Weights')
   print Weights1.name
   print Weights2.name
    # 可以看到这两个引用名称指向的是同一个内存对象
    '''
    with tf.variable_scope('word') as scope:
      
      
      
      '''
      tf.reshape:有着重新设置过形状的张量
      tensor：输入的张量
      shape：表示重新设置的张量形状，必须是int32或int64类型
      name：表示这个op名字，在tensorboard中才会用
      '''
      word_rnn_inputs = tf.reshape(
        self.embedded_inputs,
        [-1, self.max_num_words, self.emb_size]
      )
      sentence_lengths = tf.reshape(self.sentence_lengths, [-1])

      # word encoder     rnn.GRUCell应该就是一种编码方法
      cell_fw = rnn.GRUCell(self.hidden_dim)
      cell_bw = rnn.GRUCell(self.hidden_dim)
       
        
       '''
       tile()函数是用来对张量(Tensor)进行扩展的，其特点是对当前张量内的数据进行一定规则的复制。最终的输出张量维度不变。
       multiples是扩展方法  
       具体看这个网：https://blog.csdn.net/tsyccnh/article/details/82459859
       '''
      init_state_fw = tf.tile(tf.get_variable('init_state_fw',
                                              shape=[1, self.hidden_dim],
                                              initializer=tf.constant_initializer(1.0)),
                              multiples=[get_shape(word_rnn_inputs)[0], 1])
      init_state_bw = tf.tile(tf.get_variable('init_state_bw',
                                              shape=[1, self.hidden_dim],
                                              initializer=tf.constant_initializer(1.0)),
                              multiples=[get_shape(word_rnn_inputs)[0], 1])

      word_rnn_outputs, _ = bidirectional_rnn(
        cell_fw=cell_fw,
        cell_bw=cell_bw,
        inputs=word_rnn_inputs,
        input_lengths=sentence_lengths,
        initial_state_fw=init_state_fw,
        initial_state_bw=init_state_bw,
        scope=scope
      )

      self.word_outputs, self.word_att_weights = text_attention(inputs=word_rnn_outputs,
                                                                att_dim=self.att_dim,
                                                                sequence_lengths=sentence_lengths)
'''
tf.nn.dropout()是tensorflow里面为了防止或减轻过拟合而使用的函数，它一般用在全连接层
Dropout就是在不同的训练过程中随机扔掉一部分神经元。也就是让某个神经元的激活值以一定的概率p，
让其停止工作，这次训练过程中不更新权值，也不参加神经网络的计算。但是它的权重得保留下来（只是暂时
不更新而已），因为下次样本输入时它可能又得工作了
'''
      self.word_outputs = tf.nn.dropout(self.word_outputs, keep_prob=self.dropout_keep_prob)

  def _init_sent_encoder(self):
    with tf.variable_scope('sentence') as scope:
      sentence_rnn_inputs = tf.reshape(self.word_outputs, [-1, self.max_num_sents, 2 * self.hidden_dim])

      # sentence encoder
      cell_fw = rnn.GRUCell(self.hidden_dim)
      cell_bw = rnn.GRUCell(self.hidden_dim)

      init_state_fw = tf.tile(tf.get_variable('init_state_fw',
                                              shape=[1, self.hidden_dim],
                                              initializer=tf.constant_initializer(1.0)),
                              multiples=[get_shape(sentence_rnn_inputs)[0], 1])
      init_state_bw = tf.tile(tf.get_variable('init_state_bw',
                                              shape=[1, self.hidden_dim],
                                              initializer=tf.constant_initializer(1.0)),
                              multiples=[get_shape(sentence_rnn_inputs)[0], 1])

      sentence_rnn_outputs, _ = bidirectional_rnn(
        cell_fw=cell_fw,
        cell_bw=cell_bw,
        inputs=sentence_rnn_inputs,
        input_lengths=self.document_lengths,
        initial_state_fw=init_state_fw,
        initial_state_bw=init_state_bw,
        scope=scope
      )

      self.sentence_outputs, self.sent_att_weights, self.img_att_weights = visual_aspect_attention(
          text_input=sentence_rnn_outputs,
          visual_input=self.images,
          att_dim=self.att_dim,
          sequence_lengths=self.document_lengths
        )

      self.sentence_outputs = tf.nn.dropout(self.sentence_outputs, keep_prob=self.dropout_keep_prob)

  def _init_classifier(self):
    with tf.variable_scope('classifier'):
      
      
      '''
      tf.layers.dense全连接层：通常在CNN的尾部进行重新拟合，减少特征信息的损失，相当于添加一个层
      inputs：输入该网络层的数据
      units：输出的维度大小，改变inputs的最后一维
      '''
      self.logits = tf.layers.dense(
        inputs=self.sentence_outputs,
        units=self.num_classes,
        name='logits'
      )

  def get_feed_dict(self, reviews, images, labels, dropout_keep_prob=1.0):
    norm_docs, doc_sizes, sent_sizes, max_num_sents, max_num_words = batch_review_normalize(reviews)
    fd = {
      self.documents: norm_docs,
      self.document_lengths: doc_sizes,
      self.sentence_lengths: sent_sizes,
      self.max_num_sents: max_num_sents,
      self.max_num_words: max_num_words,
      self.images: batch_image_normalize(images, self.num_images),
      self.labels: labels,
      self.dropout_keep_prob: dropout_keep_prob
    }
    return fd
