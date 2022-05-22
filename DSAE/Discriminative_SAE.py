import numpy as np
import DSAE.utils as utils
import DSAE.noise as dropout
import tensorflow as tf
import scipy.sparse as sp
import math

allowed_activations = ['sigmoid', 'tanh', 'softmax', 'relu', 'linear']   
allowed_noises = [None]   
allowed_losses = ['rmse', 'cross-entropy']   

class Discriminative_SAE:
    
    def assertions(self):    
        global allowed_activations, allowed_noises, allowed_losses
        assert self.loss in allowed_losses,                                                  'Incorrect loss'                               
        assert 'list' in str(type(self.dims)),                                               'dims is a list'                           
        assert len(self.epoch) == len(self.dims),                                            'each dim must have a corresponding epoch'                        
        assert len(self.activations) == len(self.dims),                                      'each dim must have a corresponding activation'                             
        assert all(True if x > 0 else False for x in self.epoch),                            'epoch must greater than 0'                   
        assert set(self.activations + allowed_activations) == set(allowed_activations),      'Incorrect activation'
        assert utils.noise_validator(self.noise, allowed_noises),                            'Incorrect noise'        

    def __init__(self, dims, activations, Adj, epoch=1000, noise=None, loss='rmse',   
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
        self.Adj = Adj   
 
    def fit(self, x_drop, x_loss):  
        print('Layer 1 optimizing ...', flush=True)
        temp = np.copy(x_drop)
         # self.noise非None时：堆叠降噪自编码器
        out1 = self.run(data_x = (x_drop if self.noise is None else dropout.noise(self.noise, temp)),  
                        data_x_ = x_loss,
                        activation = self.activations[0],
                        hidden_dim = self.dims[0],
                        epoch = self.epoch[0], 
                        loss = self.loss,
                        batch_size = self.batch_size,
                        lr = self.lr,
                        print_step = self.print_step,
                        Adj = self.Adj)
        tf.nn.dropout(out1, 0.1)

        print('Layer 2 optimizing ...', flush=True)
        temp = np.copy(out1)
        out2 = self.run(data_x = (out1 if self.noise is None else dropout.noise(self.noise, temp)),   
                        data_x_ = out1, 
                        activation = self.activations[1],
                        hidden_dim = self.dims[1],
                        epoch = self.epoch[1], 
                        loss = self.loss,
                        batch_size = self.batch_size,
                        lr = self.lr,
                        print_step = self.print_step,
                        Adj = self.Adj)

        return out2


    def predict(self, data):
        tf.reset_default_graph()
        sess = tf.Session()
        x = tf.constant(data, dtype=tf.float32)

        weight_enc = tf.constant(self.weights[0], dtype=tf.float32)
        weight_dec = tf.constant(self.weights_dec[0], dtype=tf.float32)
        bias_enc = tf.constant(self.biases[0], dtype=tf.float32)
        bias_dec = tf.constant(self.biases_dec[0], dtype=tf.float32)

        encode = tf.matmul(x, weight_enc) + bias_enc
        encode = self.activate(encode, self.activations[0])     
        decode = tf.matmul(encode, weight_dec) + bias_dec      

        return decode.eval(session=sess)   


    def run(self, data_x, data_x_, hidden_dim, activation, loss, lr,  
            print_step, epoch, Adj, batch_size=100):
        tf.reset_default_graph()      
        input_dim = len(data_x[0])
        sess = tf.Session()         
        x = tf.placeholder(dtype=tf.float32, shape=[None, input_dim], name='x')
        x_ = tf.placeholder(dtype=tf.float32, shape=[None, input_dim], name='x_')

        encode = {
            'weights': tf.Variable(tf.random_normal([input_dim, hidden_dim])),
            'biases': tf.Variable(tf.random_normal([hidden_dim]))
        }
        
        decode = {
            'weights': tf.Variable(tf.random_normal([hidden_dim, input_dim])),  
            'biases':  tf.Variable(tf.random_normal([input_dim]))
        }

        encoded = self.activate(tf.add(tf.matmul(x, encode['weights']),encode['biases']), activation) 
        decoded = tf.add(tf.matmul(encoded, decode['weights']),decode['biases'])       
        encoded_norm = tf.reduce_sum(tf.square(encoded), 1, keepdims=True)  
        wb_norm = tf.square(encode['weights']) + tf.square(encode['biases'])

        Adj_p = tf.placeholder(dtype=tf.float32, shape=[None, None], name='Adj')
        
        L_2nd = tf.reduce_sum(
            Adj_p * (
                    encoded_norm - 2 * tf.matmul(
                        encoded, tf.transpose(encoded)
                    ) + tf.transpose(encoded_norm)
            )
        )

      
        if loss == 'rmse':
            L_1st = tf.reduce_sum(tf.square(decoded - x_))
            alpha = 1
            loss =L_1st + alpha * L_2nd 
            
        elif loss == 'cross-entropy':
            loss = -tf.reduce_mean(x_ * tf.log(decoded))

        optimizer = tf.train.RMSPropOptimizer(lr).minimize(loss)
        

        sess.run(tf.global_variables_initializer())  
        for i in range(epoch):  
            b_x, b_x_, Adj_ = utils.get_batch(data_x, data_x_, Adj, batch_size) 
            sess.run(optimizer, feed_dict={x: b_x, x_: b_x_, Adj_p: Adj_})  
            
            if (i + 1) % print_step == 0:   
                l = sess.run([loss, L_1st, L_2nd], feed_dict={x: data_x, x_: data_x_, Adj_p: Adj})  
                print('epoch {0}: global loss = {1}'.format(i, l[0]/10000), flush=True)   
                e = sess.run([encoded, encoded_norm, wb_norm],  feed_dict={x: data_x})
                
        self.weights.append(sess.run(encode['weights']))  
        self.biases.append(sess.run(encode['biases']))
        self.weights_dec.append(sess.run(decode['weights']))
        self.biases_dec.append(sess.run(decode['biases']))
        return sess.run(encoded, feed_dict={x: data_x_})  

    def activate(self, linear, name):   
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