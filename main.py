import os
import tensorflow as tf
import pprint
from g_model import seg_GAN
os.environ['CUDA_VISIBLE_DEVICES'] = "2"
pp = pprint.PrettyPrinter()

flags = tf.app.flags
flags.DEFINE_boolean("is_train", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("adverarial", False, "Adversarial or normal [50000]")
flags.DEFINE_integer("iterations", 500000, "Epoch to train [50000]")
flags.DEFINE_float("learning_rate", 1e-8, "Learning rate of for SGD [1e-8]")
flags.DEFINE_integer("batch_size", 10, "The size of batch images [10]")
flags.DEFINE_integer("show_every", 10, "The size of batch images [10]")
flags.DEFINE_integer("save_every", 1000, "save every [1000]")
flags.DEFINE_integer("test_every", 2000, "test every [5000] iterations the subject")
flags.DEFINE_integer("lr_step", 30000, "The step to decrease lr [lr_step]")
flags.DEFINE_integer("sizeCT", 512, "The size of MR patch [512]")
flags.DEFINE_float("wd", 0.0005, "weight decay [0.0005] ")
flags.DEFINE_float("lam_dice", 0.0, "weight dice loss [1.0] ")
flags.DEFINE_float("lam_fcn", 1.0, "weight fcn loss [1.0] ")
flags.DEFINE_float("lam_adv", 1.0, "weight adv loss [1.0] ")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("dir_