
"""
    @author Roger

    This class implements a 2D FCN for the task of segmentation in CT data

"""

from __future__ import division
import os
import time
from glob import glob
import tensorflow as tf
import numpy as np
from six.moves import xrange

from utils import *
from loss_functions import *
from scipy.misc import imsave
import collections
import datetime

import Queue, threading

class seg_GAN(object):
    def __init__(self, sess, batch_size=10, height=512,width=512, wd=0.0005, checkpoint_dir=None, path_patients_h5=None, learning_rate=2e-8,lr_step=30000,
                 lam_dice=1, lam_fcn=1, lam_adv=1,adversarial=False):


        self.sess = sess
        self.adversarial=adversarial
        self.lam_dice=lam_dice
        self.lam_fcn=lam_fcn
        self.lam_adv=lam_adv
        self.lr_step=lr_step
        self.wd=wd
        self.learning_rate=learning_rate
        self.batch_size=batch_size       
        self.height=height
        self.width=width
        self.checkpoint_dir = checkpoint_dir
        self.data_queue = Queue.Queue(100) # a queue with two space for 20 "chunks"
        self.path_patients_h5=path_patients_h5
        #self.data_generator = Generator_2D_slices_h5(path_patients_h5,self.batch_size)
        self.build_model()

    def build_model(self):

        self.classweights=tf.transpose(tf.constant([[1.0,1.0,1.0,1.0,1.0]],dtype=tf.float32,name='classweights'))
        self.num_classes=5

        self.inputCT=tf.placeholder(tf.float32, shape=[None, self.height, self.width, 1])#5 chans input
        #print 'inputCT shape ', self.inputCT.get_shape()
        self.CT_GT=tf.placeholder(tf.int32, shape=[None, self.height, self.width])
        batch_size_tf = tf.shape(self.inputCT)[0]  #variable batchsize so we can test here
        self.train_phase = tf.placeholder(tf.bool, name='phase_train')
        self.G, self.layer = self.generator(self.inputCT,batch_size_tf)
        print 'G shape ',self.G.get_shape
        self.prediction=tf.argmax(self.G,3)#preds by the generator
        t_vars = tf.trainable_variables()

        if self.adversarial:
            self.probs_G=tf.nn.softmax(self.G)
            self.GT_1hot=tf.one_hot(self.CT_GT,self.num_classes,1.0,0.0,axis=3,dtype=tf.float32)
            print 'GT_1hot shape ',self.GT_1hot.get_shape()
            print 'prediction shape ',self.prediction.get_shape
            self.D, self.D_logits = self.discriminator(self.GT_1hot)#real CT GT data (1hot so they have same n channels)as input      
            self.D_, self.D_logits_ = self.discriminator(self.probs_G, reuse=True)#fake generated CT probmaps as input
            self.d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.D_logits, tf.ones_like(self.D)))
            self.d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.D_logits_, tf.zeros_like(self.D_)))