import tensorflow as tf
import numpy as np

class Encoder(object):

    def __init__(self, is_training, drop_prob = None):
        self.is_training = is_training
        self.dropout = drop_prob

    def __call__(self, is_training = False, drop_prob = None):
        self.is_training = is_training
        self.dropout = drop_prob

    def __dropout__(self, x):
        if self.dropout:
            return tf.layers.dropout(x, rate = self.dropout, training = self.is_training)
        else:
            return x

    def __mask__(self, mask):
        mask_embedding = tf.constant([[0,0,0], [1,0,0], [0,1,0], [0,0,1]], dtype=np.float32)
        return tf.nn.embedding_lookup(mask_embedding, mask)

    def __pooling__(self, x, max_length, hidden_size):
        x = tf.reshape(x, [-1, max_length, hidden_size])
        x = tf.reduce_max(x, axis = 1)
        return tf.reshape(x, [-1, hidden_size])
    def __cnn_cell__(self, x, hidden_size, kernel_size, stride_size):
        x = tf.expand_dims(x, axis=1)
        x = tf.layers.conv2d(inputs=x,
                             filters = hidden_size,
                             kernel_size = [1, kernel_size],
                             strides = [1, stride_size],
                             padding = 'same',
                             kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d())
        return x
    
    def cnn(self, x, hidden_size, mask, kernel_size = 3, stride_size = 1, activation=tf.nn.relu):
        with tf.name_scope("cnn"):
            max_length = x.get_shape()[1]
            x = self.__cnn_cell__(x, hidden_size, kernel_size, stride_size)
            x = self.__pooling__(x, max_length, hidden_size)
            x = activation(x)
            return self.__dropout__(x)

