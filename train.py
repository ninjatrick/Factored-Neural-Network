import tensorflow as tf
import numpy as np
import time
import datetime
import os
import network
import json
from sklearn.metrics import average_precision_score
import sys

tf.app.flags.DEFINE_string('export_path','./data','path to data')

config_file = open(os.path.join('data', "config"), 'r')
config = json.loads(config_file.read())
config_file.close()

tf.app.flags.DEFINE_integer('max_length', config['fixlen'], 'maximum of number of words in one sentence')
tf.app.flags.DEFINE_integer('pos_num', config['maxlen'] * 2 + 1, 'number of position embedding vectors')
tf.app.flags.DEFINE_integer('num_classes', len(config['relation2id']),'maximum of relations')

tf.app.flags.DEFINE_integer('hidden_size',230,'hidden feature size')
tf.app.flags.DEFINE_integer('pos_size',5,'position embedding size')
tf.app.flags.DEFINE_integer('word_size', 50, 'word embedding size')
tf.app.flags.DEFINE_integer('POS_size', 10, 'POS embedding size')

tf.app.flags.DEFINE_integer('max_epoch',10,'maximum of training epochs')
tf.app.flags.DEFINE_integer('batch_size',64,'entity numbers used each training time')
tf.app.flags.DEFINE_float('learning_rate',0.5,'entity numbers used each training time')
tf.app.flags.DEFINE_float('weight_decay',0.000001,'weight_decay')
tf.app.flags.DEFINE_float('drop_prob',0.5,'dropout rate')

tf.app.flags.DEFINE_string('checkpoint_dir', './checkpoint/', 'path to store checkpoint')
tf.app.flags.DEFINE_string('summary_dir', './summary', 'path to store summary_dir')
tf.app.flags.DEFINE_string('test_result_dir', './test_result', 'path to store the test results')
tf.app.flags.DEFINE_integer('save_epoch', 2, 'save the checkpoint after how many epoches')

tf.app.flags.DEFINE_string('model_name', 'cnn', 'model\'s name')
tf.app.flags.DEFINE_string('pretrain_model', 'None', 'pretrain model')
FLAGS = tf.app.flags.FLAGS

from framework import Framework 
def main(_):
    from model.cnn_att import cnn_att
    model = locals()[FLAGS.model_name]
    model(is_training=True)

if __name__ == "__main__":
    tf.app.run() 
