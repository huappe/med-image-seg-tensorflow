
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