import tensorflow as tf
import numpy as np



def loss_dice(logits, labels, num_classes,batch_size_tf):
    """Calculate the loss from the logits and the labels.
    Args:
      logits: tensor, float - [batch_size, width, height, num_classes].
          Use vgg_fcn.up as logits.
      labels: Labels tensor, int32 - [batch_size, width, height].
          The ground truth of your data.
      weights: numpy array - [num_classes]
          Weighting the loss of each class
          Optional: Prioritize some classes
    Returns:
      loss: Loss tensor of type float.
    """
    #labels=tf.squeeze(labels)
    with tf.name_scope('loss'):
        #shapelables=labels.get_shape().as_list()
        probs=tf.nn.softmax(logits)        
        y_onehot=tf.one_hot(labels,num_classes,1.0,0.0,axis=3,dtype=tf.float32)
        print 'probs shape ', probs.get_shape()
        print 'y_onehot shape ', y_onehot.get_shape()
