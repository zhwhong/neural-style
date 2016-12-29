# Copyright (c) 2015-2016 Anish Athalye. Released under GPLv3.
import vgg
import pgg2
import tensorflow as tf
import numpy as np
from deal_image import *
import time
from argparse import ArgumentParser

ckpt_dir = "./ckpt-dir"
mean_pixel = [123.68,116.779,103.939]
def build_parser():
    parser = ArgumentParser()
    parser.add_argument('--content',
            dest='content', help='content image',
            metavar='CONTENT', default = 'content.jpg')
    parser.add_argument('--output',
            dest='output', help='output path',
            metavar='OUTPUT', default = 'output.jpg')
    return parser

def generator(content_name):
    content_image = tf.placeholder('float', shape = (1,256,256,3))
    YZ = tf.placeholder('float', shape=(1, 256, 256, 6))
    image = pgg2.net(YZ)
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        ckpt = tf.train.get_checkpoint_state(ckpt_dir)
        if ckpt and ckpt.model_checkpoint_path:
            print ckpt.model_checkpoint_path
            saver.restore(sess, ckpt.model_checkpoint_path)

        content = imread(content_name)
        content_pre = np.array([vgg.preprocess(content, mean_pixel)])
        print 'content_pre shape: ', content_pre.shape

        y = np.reshape(content_pre, (1, 256, 256, 3))
        z = np.random.normal(size=[1, 256, 256, 3])
        yz = np.append(y, z, axis=3)

        feed_dict = {
            YZ: yz, content_image: content_pre
        }
        images = image.eval(feed_dict = feed_dict)
        out = vgg.unprocess(images.reshape([256,256,3]),mean_pixel)
        imsave('./output/'+content_name, out)


parser = build_parser()
options = parser.parse_args()
generator(options.content)

'''
for i in range(10):
    start = time.time()
    generator(content_name)
    end = time.time()
    print end - start
'''
