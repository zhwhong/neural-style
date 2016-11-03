import cPickle
import glob
import os

import tensorflow as tf

import image_utils
import vgg

_dshape = [512, 512, 3]
_STYLE_LAYERS = ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1']
_CONTENT_LAYER = 'relu4_2'


def calc_style_features(style_dir):
    x = tf.placeholder(tf.float32,
                       shape=[None, _dshape[0], _dshape[1], _dshape[2]])
    feats = vgg.net(x)

    with tf.Session() as sess:
        for file_path in glob.glob(style_dir + "/*"):
            filename = os.path.basename(file_path)
            image = image_utils.preprocess_image(file_path, dshape=_dshape)
            image = image.reshape([1, _dshape[0], _dshape[1], _dshape[2]])
            features = sess.run(feats, feed_dict={x: image})
            values = {}
            for layer in _STYLE_LAYERS:
                values[layer] = features[layer]
            pkl_file = 'data/features/%s.pkl' % filename
            with open(pkl_file, 'w') as f:
                cPickle.dump(values, f)


def calc_content_features(content_dir):
    x = tf.placeholder(tf.float32,
                       shape=[None, _dshape[0], _dshape[1], _dshape[2]])
    feats = vgg.net(x)

    with tf.Session() as sess:
        for file_path in glob.glob(content_dir + "/*"):
            filename = os.path.basename(file_path)
            image = image_utils.preprocess_image(file_path, dshape=_dshape)
            image = image.reshape([1, _dshape[0], _dshape[1], _dshape[2]])
            features = sess.run(feats, feed_dict={x: image})
            values = features[_CONTENT_LAYER]
            pkl_file = 'data/features/%s.pkl' % filename
            with open(pkl_file, 'w') as f:
                cPickle.dump(values, f)


def main(style_dir, content_dir):
    calc_style_features(style_dir)
    calc_content_features(content_dir)


if __name__ == '__main__':
    style_dir = 'data/styles'
    content_dir = 'data/hair'
    main(style_dir, content_dir)