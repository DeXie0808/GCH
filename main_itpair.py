import os
from setting_old import *
from GH_itpair import GH
import tensorflow as tf
os.environ['CUDA_VISIBLE_DEVICES'] = select_gpu

if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
# if not os.path.exists(sample_dir):
#     os.makedirs(sample_dir)
# if not os.path.exists(test_dir):
#     os.makedirs(test_dir)


gpuconfig = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=per_process_gpu_memory_fraction))

with tf.Session(config=gpuconfig) as sess:
    model = GH(sess)
    model.Train() if phase == 'train' else model.test()
