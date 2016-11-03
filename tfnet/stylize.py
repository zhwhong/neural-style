import argparse
import importlib
import json
import logging
import os

import tensorflow as tf

from filter import filter_image
from image_utils import *

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s",
                    level=logging.INFO)

_image_size = 960

_DSHAPE = [_image_size, _image_size, 3]
_MEAN_PIXEL = [123.68, 116.779, 103.939]

_ROOT = os.path.realpath(os.path.dirname(__file__))
_MODELS = os.path.join(_ROOT, 'models')


def get_parser():
    p = argparse.ArgumentParser()
    p.add_argument('-i', '--input_image', type=str, required=True,
                   help='image to be styled')
    p.add_argument('-o', '--output_path', type=str, required=True,
                   help='output image path')
    p.add_argument('-s', '--style_num', type=int, default=0,
                   help='style name')
    p.add_argument('-g', '--generator', type=str, default='',
                   help='generator name')
    p.add_argument('-m', '--model', type=str, default='', help='model')
    p.add_argument('-d', '--device', type=str, default='/cpu',
                   help='device')
    return p


def gen(input_path, output_path, config, device='/cpu'):
    generator_name = config['gen'] if config['gen'] is not None else 'g1'
    style = os.path.join(_ROOT, config['ckpt'])
    logging.info("Style checkpoint: %s" % style)
    assert os.path.isfile(style), 'Error: style checkpoint not found!'
    noise_rate = (0.1
                  if 'noise_rate' not in config.keys()
                  else config.get('noise_rate'))
    style_rate = (0.7
                  if 'style_rate' not in config.keys()
                  else config.get('style_rate'))

    content_shape = [1, ] + _DSHAPE
    content = imread(input_path)
    content_np = crop_image(content, _DSHAPE)
    content_np = filter_image(content_np, contrast=1.2, color=1.2)

    with tf.device(device):
        x = tf.placeholder('float', shape=[1, _image_size, _image_size, 6])
        logging.info("Generator: %s" % generator_name)
        generator_class = importlib.import_module('gens.' + generator_name)
        generator = generator_class.Generator(x, _DSHAPE)
        net = generator.net()
    saver = tf.train.Saver(generator.variables)
    with tf.Session() as sess:
        saver.restore(sess, style)
        y = np.array([content_np - _MEAN_PIXEL])
        y = np.reshape(y, content_shape)
        z = np.random.normal(scale=noise_rate, size=content_shape)
        yz = np.append(y, z, axis=3)
        image = sess.run(net, feed_dict={x: yz})
    image = image.reshape(_DSHAPE) + _MEAN_PIXEL
    image = style_rate * image + (1 - style_rate) * content_np
    if output_path is not None:
        imsave(output_path, image)
    return image


def process(input_path, output_path=None, s_num=0):
    with open(os.path.join(_ROOT, 'style_config.json'), 'r') as sf:
        s_config = json.load(sf)
    s_num = s_num if s_num < len(s_config) else 0
    return gen(input_path=input_path,
               output_path=output_path,
               config=s_config[s_num])


if __name__ == '__main__':
    parser = get_parser()
    options = parser.parse_args()

    # get config
    if len(options.generator) > 0 and len(options.model) > 0:
        conf = {'gen': options.generator, 'ckpt': options.model}
    else:
        with open(os.path.join(_ROOT, 'style_config.json'), 'r') as f:
            style_config = json.load(f)
        style_num = options.style_num \
            if options.style_num < len(style_config) else 0
        conf = style_config[style_num]

    func = 'gen' if 'type' not in conf.keys() else conf.get('type')
    globals()[func](input_path=options.input_image,
                    output_path=options.output_path,
                    config=conf,
                    device=options.device)
