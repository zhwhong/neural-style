import tensorflow as tf
import tflearn
from tflearn.layers.conv import avg_pool_2d

_act = tflearn.activations.relu
_norm = tflearn.local_response_normalization


def net(input_image, layers=6):
    # ratios = [32, 16, 8, 4, 2, 1]
    ratios = [pow(2, layers - i - 1) for i in range(layers)]
    conv_num = 8
    network = []
    for i in range(len(ratios)):
        network.append(avg_pool_2d(input_image, ratios[i]))

        # block_i_0, block_i_1, block_i_2
        for block in range(3):
            with tf.name_scope('block_%d_%d' % (i, block)):
                filter_size = 1 if (block + 1) % 3 == 0 else 3
                network[i] = tflearn.conv_2d(network[i], nb_filter=conv_num,
                                             filter_size=filter_size,
                                             weights_init='xavier',
                                             name='Conv_%d_%d' % (i, block))
                network[i] = _norm(network[i])
                # network[i] = tf.nn.batch_normalization(network[i], 0, 1.0,
                #                                        0.0, 1.0, 1e-5,
                #                                        'BatchNorm')
                network[i] = _act(network[i])

        if i == 0:
            network[i] = tflearn.upsample_2d(network[i], 2)
        else:
            upnet = _norm(network[i - 1])
            downnet = _norm(network[i])
            # upnet = tf.nn.batch_normalization(network[i - 1], 0, 1.0, 0.0, 1.0,
            #                                   1e-5, 'BatchNorm')
            # downnet = tf.nn.batch_normalization(network[i], 0, 1.0, 0.0, 1.0,
            #                                     1e-5, 'BatchNorm')
            # join_i
            network[i] = tflearn.merge([upnet, downnet], 'concat', axis=3)
            # block_i_3, block_i_4, block_i_5
            for block in range(3, 6):
                with tf.name_scope('block_%d_%d' % (i, block)):
                    filter_size = 1 if (block + 1) % 3 == 0 else 3
                    network[i] = tflearn.conv_2d(network[i],
                                                 nb_filter=conv_num * i,
                                                 filter_size=filter_size,
                                                 weights_init='xavier',
                                                 name='Conv_%d_%d' % (i, block))
                    network[i] = _norm(network[i])
                    # network[i] = tf.nn.batch_normalization(network[i], 0, 1.0,
                    #                                        0.0, 1.0, 1e-5,
                    #                                        'BatchNorm')
                    network[i] = _act(network[i])

            if i != len(ratios) - 1:
                network[i] = tflearn.upsample_2d(network[i], 2)

    network[len(ratios) - 1] = tflearn.conv_2d(network[len(ratios) - 1],
                                               nb_filter=3,
                                               filter_size=1,
                                               weights_init='xavier',
                                               name='Conv2d_out')
    return network[len(ratios) - 1]