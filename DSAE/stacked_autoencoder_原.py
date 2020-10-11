import numpy as np
import deepautoencoder.utils as utils
import tensorflow as tf

allowed_activations = ['sigmoid', 'tanh', 'softmax', 'relu', 'linear']   # 激活函数
allowed_noises = [None, 'gaussian', 'mask','impute']   # 可选的 噪音类型
allowed_losses = ['rmse', 'cross-entropy']   # 可选的 损失函数


class StackedAutoEncoder:
    """A deep autoencoder with denoising capability"""

    def assertions(self):    # 断言：参数正确性判断
        global allowed_activations, allowed_noises, allowed_losses
        assert self.loss in allowed_losses,                                             'Incorrect loss given'
        assert 'list' in str(type(self.dims)),                                          'dims must be a list even if there is one layer.'
        assert len(self.epoch) == len(self.dims),                                       "No. of epochs must equal to no. of hidden layers"
        assert len(self.activations) == len(self.dims),                                 "No. of activations must equal to no. of hidden layers"
        assert all(True if x > 0 else False for x in self.epoch),                       "No. of epoch must be at least 1"
        assert set(self.activations + allowed_activations) == set(allowed_activations), "Incorrect activation given."
        assert utils.noise_validator(self.noise, allowed_noises),                       "Incorrect noise given"

    def __init__(self, dims, activations, epoch=1000, noise=None, loss='rmse',   # 初始化
                 lr=0.001, batch_size=100, print_step=50):
        self.print_step = print_step
        self.batch_size = batch_size
        self.lr = lr
        self.loss = loss
        self.activations = activations
        self.noise = noise
        self.epoch = epoch
        self.dims = dims
        self.assertions()
        self.depth = len(dims)
        self.weights, self.biases = [], []
        self.weights_dec, self.biases_dec = [], []

    def add_noise(self, x):     ## TODO: 加噪音
        if self.noise == 'impute':
            return x
        if self.noise == 'gaussian':
            n = np.random.normal(0, 0.1, (len(x), len(x[0])))
            return x + n
        if 'mask' in self.noise:
            frac = float(self.noise.split('-')[1])
            temp = np.copy(x)
            for i in temp:
                n = np.random.choice(len(i), round(frac * len(i)), replace=False)
                i[n] = 0
            return temp
        if self.noise == 'sp':
            pass

    def fit(self, x):   ## TODO: 对数据拟合
        for i in range(self.depth):
            print('Layer {0}'.format(i + 1))
            if self.noise is None:
                x = self.run(data_x  = x, 
                             data_x_ = x,
                             activation = self.activations[i],
                             hidden_dim = self.dims[i], 
                             epoch = self.epoch[i], 
                             loss = self.loss,
                             batch_size = self.batch_size, 
                             lr = self.lr,
                             print_step = self.print_step)
            else:
                temp = np.copy(x)
                x = self.run(data_x = self.add_noise(temp),   # 调用 add_noise()函数调加噪音，后再 run()
                             data_x_ = x,
                             activation = self.activations[i], 
                             hidden_dim = self.dims[i],
                             epoch = self.epoch[i], 
                             loss = self.loss,
                             batch_size = self.batch_size,
                             lr = self.lr, 
                             print_step = self.print_step)

    def transform(self, data):  ## TODO: 编码部分
        tf.reset_default_graph()
        sess = tf.Session()
        x = tf.constant(data, dtype=tf.float32)
        for w, b, a in zip(self.weights, self.biases, self.activations):
            weight = tf.constant(w, dtype=tf.float32)
            bias = tf.constant(b, dtype=tf.float32)
            layer = tf.matmul(x, weight) + bias
            x = self.activate(layer, a)
        return x.eval(session=sess)
    
    def decode_(self, decode):   ## TODO: 解码部分
        tf.reset_default_graph()
        sess = tf.Session()
        x = tf.constant(decode, dtype=tf.float32)
        for w, b in zip(self.weights_dec, self.biases_dec):
            weight = tf.constant(w, dtype=tf.float32)
            bias = tf.constant(b, dtype=tf.float32)
            layer = tf.matmul(x, weight) + bias
            return sess.run(layer)
        return sess.run(layer)  

    def fit_transform(self, x):   ## TODO: 先拟合训练模型，再编码
        self.fit(x)
        return self.transform(x)

    def run(self, data_x, data_x_, hidden_dim, activation, loss, lr,   ## TODO: 
            print_step, epoch, batch_size=100):
        tf.reset_default_graph()      # 调tensflow中的相关函数，【待学】
        input_dim = len(data_x[0])
        sess = tf.Session()          ### FIXME:
        x = tf.placeholder(dtype=tf.float32, shape=[None, input_dim], name='x')
        x_ = tf.placeholder(dtype=tf.float32, shape=[None, input_dim], name='x_')

        encode = {
            'weights': tf.Variable(tf.truncated_normal([input_dim, hidden_dim], dtype=tf.float32)),
            'biases': tf.Variable(tf.truncated_normal([hidden_dim], dtype=tf.float32))
        }
        
        decode = {
            'biases': tf.Variable(tf.truncated_normal([input_dim], dtype=tf.float32)),  
            'weights': tf.transpose(encode['weights'])
        }

        ## TODO: 编码：缺失矩阵X * 编码权重矩阵 + 编码偏置矩阵  → 激活函数
        encoded = self.activate(tf.matmul(x, encode['weights']) + encode['biases'], activation)
        ## TODO: 解码：编码矩阵encoded * 解码权重矩阵 + 解码偏置矩阵  
        decoded = tf.matmul(encoded, decode['weights']) + decode['biases']

        # 定义损失函数reconstruction loss   计算重构损失： x_真实值、decoded：重构值
        if loss == 'rmse':
            loss = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(x_, decoded))))
        elif loss == 'cross-entropy':
            loss = -tf.reduce_mean(x_ * tf.log(decoded))
        # 定义优化器
        train_op = tf.train.AdamOptimizer(lr).minimize(loss)


        sess.run(tf.global_variables_initializer())   ###
        for i in range(epoch):  ## 3000
            b_x, b_x_ = utils.get_batch(data_x, data_x_, batch_size)
            sess.run(train_op, feed_dict={x: b_x, x_: b_x_})  ###
            
            if (i + 1) % print_step == 0:
                l = sess.run(loss, feed_dict={x: data_x, x_: data_x_})  ## x:缺失矩阵，x_:真实矩阵
                print('epoch {0}: global loss = {1}'.format(i, l))   ## TODO: 输出损失信息
        # self.loss_val = l
        # debug
        # print('Decoded', sess.run(decoded, feed_dict={x: self.data_x_})[0])
        self.weights.append(sess.run(encode['weights']))  ## 计算并保存 weights权重矩阵
        self.biases.append(sess.run(encode['biases']))
        self.weights_dec.append(sess.run(decode['weights']))
        self.biases_dec.append(sess.run(decode['biases']))
        return sess.run(encoded, feed_dict={x: data_x_})  ## TODO: 调用run后返回值没用到，无意义

    def activate(self, linear, name):   # 激活函数选择 
        if   name == 'sigmoid':
            return tf.nn.sigmoid(linear, name='encoded')
        elif name == 'softmax':
            return tf.nn.softmax(linear, name='encoded')
        elif name == 'linear':
            return linear
        elif name == 'tanh':
            return tf.nn.tanh(linear, name='encoded')
        elif name == 'relu':
            return tf.nn.relu(linear, name='encoded')

