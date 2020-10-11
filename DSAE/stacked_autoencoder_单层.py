import numpy as np
import deepautoencoder.utils as utils
import deepautoencoder.add_noise as dropout
import tensorflow as tf
import scipy.sparse as sp

allowed_activations = ['sigmoid', 'tanh', 'softmax', 'relu', 'linear']   # 激活函数
allowed_noises = [None, 'gaussian', 'mask', 'deepImpute']   # 可选的 噪音类型
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
                 lr=0.001, batch_size=100, print_step=50, masked_idx=None):
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
        self.decoded_layer = None
    
    def fit(self, x_drop, x_true):   ## TODO: 对数据拟合 x：data_raw_process、
        for i in range(self.depth):
            print('Layer {0}'.format(i + 1))
            if self.noise is None:  ## FIXME: 未指定 noise参数
                x_drop = self.run(data_x  = x_drop, 
                             data_x_ = x_true,
                             activation = self.activations[i],
                             hidden_dim = self.dims[i], 
                             epoch = self.epoch[i], 
                             loss = self.loss,
                             batch_size = self.batch_size, 
                             lr = self.lr,
                             print_step = self.print_step)
            else:
                temp = np.copy(x_drop) 
                x_drop = self.run(data_x = dropout.add_noise(self.noise, temp),   # 调用 add_noise()函数调加噪音，后再 run()
                             data_x_ = x_true,
                             activation = self.activations[i],
                             hidden_dim = self.dims[i],
                             epoch = self.epoch[i], 
                             loss = self.loss,
                             batch_size = self.batch_size,
                             lr = self.lr, 
                             print_step = self.print_step)

    def predict(self, data):
        tf.reset_default_graph()
        sess = tf.Session()
        x = tf.constant(data, dtype=tf.float32)
                                ## (3978,400)   (3978,400)      激活函数          (400,3978)        (3978,)
        for w, b, a, wd,bd in zip(self.weights, self.biases, self.activations, self.weights_dec,self.biases_dec):
            weight_enc = tf.constant(w, dtype=tf.float32)   ##
            bias_enc   = tf.constant(b, dtype=tf.float32)   ##
            weight_dec = tf.constant(wd, dtype=tf.float32)  ##
            bias_dec   = tf.constant(bd, dtype=tf.float32)  ##
            
            layer = tf.matmul(x, weight_enc) + bias_enc
            x = self.activate(layer, a)
            output = tf.matmul(x, weight_dec) + bias_dec
        return output.eval(session=sess)

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

    def fit_transform(self, x_drop, x_true):   ## TODO: 先拟合训练模型，再编码
        self.fit(x_drop, x_true)
        return self.transform(x_drop)


    def run(self, data_x, data_x_, hidden_dim, activation, loss, lr,   ## TODO: 
            print_step, epoch, batch_size=100):
        tf.reset_default_graph()      # 调tensflow中的相关函数，【待学】
        input_dim = len(data_x[0])
        sess = tf.Session()          ### FIXME:
        x = tf.placeholder(dtype=tf.float32, shape=[None, input_dim], name='x')
        x_ = tf.placeholder(dtype=tf.float32, shape=[None, input_dim], name='x_')

        encode = {
            # 'weights': tf.Variable(tf.truncated_normal([input_dim, hidden_dim], dtype=tf.float32)), ## (3978,400)
            'weights': tf.Variable(tf.random_normal([input_dim, hidden_dim])), ## (3978,400)
            'biases': tf.Variable(tf.random_normal([hidden_dim]))  ##(400, )
        }
        
        decode = {
            'weights': tf.Variable(tf.random_normal([hidden_dim, input_dim])),  ##(400,3978)
            'biases':  tf.Variable(tf.random_normal([input_dim])) ## (3978,)
        }

        ## TODO: 编码：缺失矩阵X * 编码权重矩阵 + 编码偏置矩阵  → 激活函数
        encoded = self.activate(tf.add(tf.matmul(x, encode['weights']),encode['biases']), activation)
        ## TODO: 解码：编码矩阵encoded * 解码权重矩阵 + 解码偏置矩阵  
        decoded = tf.add(tf.matmul(encoded, decode['weights']),decode['biases'])
        # 定义损失函数reconstruction loss   计算重构损失： x_真实值、decoded：重构值

        if loss == 'rmse':
            # loss = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(x_, decoded))))
            loss = tf.contrib.losses.mean_squared_error(decoded, x_)  # 1：较好，列值差距大，PCC高

            # #? 因为目前还没有好的方法解决tf.where(express == 0)报错的问题，所以可以尝试以下方法
            # #? 1. 首先使用正常的python语句生成一个形如[[x1,y1], [x2,y2], ...]的二维数组mtx
            # #? 2. 然后在这里使用zero_idx = tf.constant(mtx)转换成tf常量形式
            # #? 3. 再tf.gather_nd(x_, zero_idx); tf.gather_nd(decoded, zero_idx)，即和167，168行一样
            # #? 结果：第一次尝试--报错  修改后--成功【不使用batch_size，而是整个数据集全部丢进去训练】
            ##  TODO: 无法使用 batch_size
            # zero_idx = np.where(data_x_ == 0)    
            # zero_idx = np.vstack((zero_idx[0], zero_idx[1])).T   ## (x1,y1)
            #                                                      ## (x2,y2)    
            # zero_idx_tf = tf.constant(zero_idx.tolist())
            # data_true_tf = tf.gather_nd(x_, zero_idx_tf)         ## 从x_真实数组中取出下标对应值(即 0值)
            # data_pred_tf = tf.gather_nd(decoded, zero_idx_tf)   ## 从decoded补全数组中取出对应补全值
            # loss = tf.contrib.losses.mean_squared_error(data_pred_tf, data_true_tf)
            # print()

            ## TODO: 可使用 batch_size 批量训练
            # zero = tf.constant(0, dtype=np.float32)
            # where = tf.equal(x_, zero)
            # zero_idx_tf = tf.where(where)
            # # zero_idx_tf = tf.where(express <= 0) ## ??报错 不能写express==0或express!=0(但可以express>0或<0)
            # data_true_tf = tf.gather_nd(x_, zero_idx_tf)
            # data_pred_tf = tf.gather_nd(decoded, zero_idx_tf)
            # loss = tf.contrib.losses.mean_squared_error(data_pred_tf, data_true_tf)
            
            
        elif loss == 'cross-entropy':
            loss = -tf.reduce_mean(x_ * tf.log(decoded))
        # 定义优化器
        # optimizer = tf.train.AdamOptimizer(lr).minimize(loss)
        optimizer = tf.train.RMSPropOptimizer(lr).minimize(loss)


        sess.run(tf.global_variables_initializer())   ###
        for i in range(epoch):  ## 3000
            b_x, b_x_ = utils.get_batch(data_x, data_x_, batch_size)
            sess.run(optimizer, feed_dict={x: b_x, x_: b_x_})  ###
            
            if (i + 1) % print_step == 0:
                l = sess.run(loss, feed_dict={x: data_x, x_: data_x_})  ## x:缺失矩阵，x_:真实矩阵
                print('epoch {0}: global loss = {1}'.format(i, l))   ## TODO: 输出损失信息
        # self.loss_val = l
        # debug
        print('Decoded', sess.run(decoded, feed_dict={x: data_x_})[0])
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

print()