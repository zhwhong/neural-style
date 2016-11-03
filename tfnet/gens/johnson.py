import logging

import numpy as np
import tensorflow as tf


def weight_variable(shape, name='Variable'):
    return tf.Variable(
        np.random.normal(scale=0.01, size=shape).astype(np.float32),
        name=name)


class Generator(object):
    def __init__(self, input_image, dshape):
        _, self._height, self._width, self._channel = map(lambda i: i.value,
                                                          input_image.get_shape())
        self._input_image = input_image
        self._conv_blocks = 2
        self._res_blocks = 5
        self._n_channel = 32
        self._variables = self._init_variables()
        self._layers = self._init_layers()

    @property
    def variables(self):
        return self._variables.values()

    def _init_layers(self):
        layers = ['base_conv_block']
        for i in range(self._conv_blocks):
            layers.append('conv_block_%d' % i)
        for i in range(self._res_blocks):
            layers.append('res_block_%d' % i)
        for i in range(self._conv_blocks):
            layers.append('deconv_block_%d' % i)
        layers.append('base_deconv_block')
        return layers

    def _init_variables(self):
        vs = {}
        n_channel = self._n_channel
        with tf.name_scope('johnson_vars'):
            vs['block_1'] = weight_variable([9, 9, self._channel, n_channel],
                                            name='block_1')
            vs['block_2'] = weight_variable([3, 3, n_channel, n_channel * 2],
                                            name='block_2')
            vs['block_3'] = weight_variable(
                [3, 3, n_channel * 2, n_channel * 4],
                name='block_3')
            for i in range(self._res_blocks):
                for j in range(2):
                    name = 'res_%d_%d' % (i, j)
                    vs[name] = weight_variable(
                        [3, 3, n_channel * 4, n_channel * 4],
                        name=name)
            vs['block_4'] = weight_variable(
                [3, 3, n_channel * 2, n_channel * 4],
                name='block_4')
            vs['block_5'] = weight_variable([3, 3, n_channel, n_channel * 2],
                                            name='block_5')
            vs['block_6'] = weight_variable([9, 9, 3, n_channel],
                                            name='block_6')
        return vs

    def _res_block(self, network, i):
        net1 = tf.nn.conv2d(network, filter=self._variables['res_%d_0' % i],
                            strides=[1, 1, 1, 1], padding='SAME')
        net1 = tf.nn.batch_normalization(net1, 0, 1.0, 0.0, 1.0, 1e-5)
        net1 = tf.nn.relu(net1)
        net1 = tf.nn.conv2d(net1, filter=self._variables['res_%d_1' % i],
                            strides=[1, 1, 1, 1], padding='SAME')
        net1 = tf.nn.batch_normalization(net1, 0, 1.0, 0.0, 1.0, 1e-5)
        return network + net1

    def _conv_block(self, network, i):
        network = tf.nn.conv2d(network, filter=self._variables['block_%d' % i],
                               strides=[1, 1, 1, 1], padding='SAME')
        network = tf.nn.batch_normalization(network, 0, 1.0, 0.0, 1.0, 1e-5)
        network = tf.nn.relu(network)
        logging.info(network.get_shape())
        return network

    def _deconv_block(self, network, i, output_shape):
        network = tf.nn.conv2d_transpose(
            network, filter=self._variables['block_%d' % i],
            output_shape=output_shape, strides=[1, 2, 2, 1])
        network = tf.nn.batch_normalization(network, 0, 1.0, 0.0, 1.0, 1e-5)
        network = tf.nn.relu(network)
        logging.info(network.get_shape())
        return network

    def net(self):
        network = self._input_image
        with tf.name_scope('johnson'):
            batch_size = tf.shape(self._input_image)[0]
            # batch_size = 1

            for i in range(3):
                with tf.name_scope('block_%d' % i):
                    network = self._conv_block(network, i)

            for i in range(5):
                with tf.name_scope('res_block_%d' % i):
                    network = self._res_block(network, i)

            with tf.name_scope('block4'):
                deconv_shape_block4 = tf.pack(
                    [batch_size, self._height / 2, self._width / 2,
                     self._n_channel * 2])
                network = self._deconv_block(network, 4, deconv_shape_block4)

            with tf.name_scope('block5'):
                deconv_shape_block5 = tf.pack(
                    [batch_size, self._height, self._width, self._n_channel])
                network = self._deconv_block(network, 5, deconv_shape_block5)
            with tf.name_scope('block6'):
                deconv_shape_block6 = tf.pack(
                    [batch_size, self._height, self._width, 3])
                network = tf.nn.conv2d_transpose(
                    network, filter=self._variables['block_6'],
                    output_shape=deconv_shape_block6,
                    strides=[1, 1, 1, 1])
                # block6 = tf.nn.batch_normalization(block6, 0, 1.0, 0.0, 1.0, 1e-5)
                logging.info(network.get_shape())
            network = tf.nn.tanh(network)
            ones = tf.constant(1.0, shape=[self._height, self._width, 3])
            output = tf.mul((network + ones), 127.5)
            logging.info(output.get_shape())
        return output