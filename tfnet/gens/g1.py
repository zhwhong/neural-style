import numpy as np
import tensorflow as tf
import tflearn

_act = tflearn.leaky_relu


def weight_variable(shape, name):
    # initial = tf.truncated_normal(shape, stddev=0.1)
    # logging.info("Var %s: %s" % (name, str(shape)))
    var = tf.Variable(
        np.random.normal(scale=0.01, size=shape).astype(np.float32),
        name=name)
    return var


class Generator(object):
    def __init__(self, input_image, dshape, layers=6):
        _, self._height, self._width, self._channel = map(lambda i: i.value,
                                                          input_image.get_shape())
        self._input_image = input_image
        self._layers = layers
        # ratios = [32, 16, 8, 4, 2, 1]
        self._ratios = [pow(2, layers - i - 1) for i in range(layers)]

        self._variables = self._init_variables()

    def _init_variables(self):
        vs = {}
        conv_num = 8
        for i in range(self._layers):
            for block in range(3):
                with tf.name_scope('block_%d_%d' % (i, block)):
                    name = 'C_%d%d' % (i, block)
                    filter_size = 1 if (block + 1) % 3 == 0 else 3
                    channel = conv_num if block > 0 else self._channel
                    vs[name] = weight_variable(
                        [filter_size, filter_size, channel, conv_num],
                        name=name)
            if i != 0:
                for block in range(3, 6):
                    with tf.name_scope('block_%d_%d' % (i, block)):
                        name = 'C_%d%d' % (i, block)
                        filter_size = 1 if (block + 1) % 3 == 0 else 3
                        vs[name] = weight_variable(
                            [filter_size, filter_size, conv_num * (i + 1),
                             conv_num * (i + 1)],
                            name=name)
        vs['C_wo'] = weight_variable([1, 1, conv_num * self._layers, 3],
                                     name='C_wo')
        return vs

    @property
    def variables(self):
        return self._variables.values()

    def net(self):
        with tf.name_scope('g1_%d' % self._layers):
            network = []
            for i in range(self._layers):
                network.append(
                    tf.nn.avg_pool(self._input_image,
                                   [1, self._ratios[i], self._ratios[i], 1],
                                   [1, self._ratios[i], self._ratios[i], 1],
                                   "SAME"))

                # block_i_0, block_i_1, block_i_2
                for block in range(3):
                    with tf.name_scope('block_%d_%d' % (i, block)):
                        network[i] = tf.nn.conv2d(
                            network[i],
                            filter=self._variables['C_%d%d' % (i, block)],
                            strides=[1, 1, 1, 1],
                            padding='SAME')
                        network[i] = tf.nn.batch_normalization(
                            network[i], 0, 1.0, 0.0, 1.0, 1e-5)
                        network[i] = _act(network[i])

                if i == 0:
                    network[i] = tflearn.upsample_2d(network[i], 2)
                else:
                    with tf.name_scope('join_%d' % i):
                        upnet = tf.nn.batch_normalization(
                            network[i - 1], 0, 1.0, 0.0, 1.0, 1e-5)
                        downnet = tf.nn.batch_normalization(
                            network[i], 0, 1.0, 0.0, 1.0, 1e-5)
                        # join_i
                        network[i] = tf.concat(3, [upnet, downnet])
                    # block_i_3, block_i_4, block_i_5
                    for block in range(3, 6):
                        with tf.name_scope('block_%d_%d' % (i, block)):
                            network[i] = tf.nn.conv2d(
                                network[i],
                                filter=self._variables['C_%d%d' % (i, block)],
                                strides=[1, 1, 1, 1],
                                padding='SAME')
                            network[i] = tf.nn.batch_normalization(
                                network[i], 0, 1.0, 0.0, 1.0, 1e-5)
                            network[i] = _act(network[i])

                    if i != len(self._ratios) - 1:
                        network[i] = tflearn.upsample_2d(network[i], 2)

            nn = self._layers - 1
            output = tf.nn.conv2d(network[nn], filter=self._variables['C_wo'],
                                  strides=[1, 1, 1, 1], padding='SAME')
        return output

        # def weight_variable(shape, name):
        #     # initial = tf.truncated_normal(shape, stddev=0.1)
        #     # logging.info("Var %s: %s" % (name, str(shape)))
        #     var = tf.Variable(
        #         np.random.normal(scale=0.01, size=shape).astype(np.float32),
        #         name=name)
        #     return var
        #
        #
        # def net(input_image, layers=6):
        #     with tf.name_scope('g1_%d' % layers):
        #         _, _, _, n_channel = map(lambda i: i.value,
        #                                  input_image.get_shape())
        #         # ratios = [32, 16, 8, 4, 2, 1]
        #         ratios = [pow(2, layers - i - 1) for i in range(layers)]
        #         conv_num = 8
        #         network = []
        #         w = [{} for _ in range(len(ratios))]
        #         for i in range(len(ratios)):
        #             network.append(tf.nn.avg_pool(input_image, [1, ratios[i], ratios[i], 1],
        #                                           [1, ratios[i], ratios[i], 1], "SAME"))
        #             # network.append(avg_pool_2d(input_image, ratios[i]))
        #
        #             # block_i_0, block_i_1, block_i_2
        #             for block in range(3):
        #                 with tf.name_scope('block_%d_%d' % (i, block)):
        #                     filter_size = 1 if (block + 1) % 3 == 0 else 3
        #                     channel = conv_num if block > 0 else n_channel
        #                     w[i][block] = weight_variable(
        #                         [filter_size, filter_size, channel, conv_num],
        #                         name='C_%d%d' % (i, block))
        #                     network[i] = tf.nn.conv2d(network[i], filter=w[i][block],
        #                                               strides=[1, 1, 1, 1], padding='SAME')
        #                     network[i] = tf.nn.batch_normalization(network[i], 0, 1.0,
        #                                                            0.0, 1.0, 1e-5)
        #                     network[i] = _act(network[i])
        #
        #             if i == 0:
        #                 network[i] = tflearn.upsample_2d(network[i], 2)
        #             else:
        #                 with tf.name_scope('join_%d' % (i)):
        #                     upnet = tf.nn.batch_normalization(network[i - 1], 0, 1.0, 0.0, 1.0,
        #                                                       1e-5, 'BatchNorm')
        #                     downnet = tf.nn.batch_normalization(network[i], 0, 1.0, 0.0, 1.0,
        #                                                         1e-5, 'BatchNorm')
        #                     # join_i
        #                     network[i] = tf.concat(3, [upnet, downnet])
        #                 # block_i_3, block_i_4, block_i_5
        #                 for block in range(3, 6):
        #                     with tf.name_scope('block_%d_%d' % (i, block)):
        #                         filter_size = 1 if (block + 1) % 3 == 0 else 3
        #                         w[i][block] = weight_variable(
        #                             [filter_size, filter_size, conv_num * (i + 1),
        #                              conv_num * (i + 1)],
        #                             name='C_%d%d' % (i, block))
        #                         network[i] = tf.nn.conv2d(network[i], filter=w[i][block],
        #                                                   strides=[1, 1, 1, 1], padding='SAME')
        #                         network[i] = tf.nn.batch_normalization(network[i], 0, 1.0,
        #                                                                0.0, 1.0, 1e-5)
        #                         network[i] = _act(network[i])
        #
        #                 if i != len(ratios) - 1:
        #                     network[i] = tflearn.upsample_2d(network[i], 2)
        #
        #         nn = len(ratios) - 1
        #         w_o = weight_variable([1, 1, conv_num * (nn + 1), 3],
        #                               name='C_wo')
        #         output = tf.nn.conv2d(network[nn], filter=w_o,
        #                               strides=[1, 1, 1, 1], padding='SAME')
        #     return output