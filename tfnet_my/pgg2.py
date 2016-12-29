import tensorflow as tf
import tflearn
from tflearn.layers.conv import max_pool_2d

'''
    To see the struct go to ./pgg.jpg
        and network.jpg for prisma's
    for more details run this file and use:
        tensorboard --logdir='logs/nn_logs'
'''

def net(input_image):
    layers = (
        ['block1_1', 8, 3], ['block1_2', 8, 3], ['block1_3', 8, 1],
        ['block2_1', 8, 3], ['block2_2', 8, 3], ['block2_3', 8, 1],
        ['block3_1', 8, 3], ['block3_2', 8, 3], ['block3_3', 8, 1],
        ['block4_1', 8, 3], ['block4_2', 8, 3], ['block4_3', 8, 1],
        ['block5_1', 8, 3], ['block5_2', 8, 3], ['block5_3', 8, 1],
        ['joinr4'],
        ['block4_4', 16, 3], ['block4_5',16, 3], ['block4_6', 16, 1],
        ['joinr3'],
        ['block3_4', 24, 3], ['block3_5',24, 3], ['block3_6', 24, 1],
        ['joinr2'],
        ['block2_4', 32, 3], ['block2_5',32, 3], ['block2_6', 32, 1],
        ['joinr1'],
        ['block1_4', 40, 3], ['block1_5',40, 3], ['block1_6', 40, 1],
        ['block1_7', 3, 1]
    )

    net = {}
    net[1] = input_image
    net[2] = max_pool_2d(net[1], 2)
    net[3] = max_pool_2d(net[2], 2)
    net[4] = max_pool_2d(net[3], 2)
    net[5] = max_pool_2d(net[4], 2)

    for layer in layers:
        name = layer[0][:5]
        rank = int(layer[0][5])
        with tf.name_scope(layer[0]):
            if name == 'block':
                out = (int(layer[0][7]) == 7)
                net[rank] = _block_layer(net[rank],layer[1],layer[2], out)
                splits = tf.split(3, layer[1], net[rank])
                for i, split in enumerate(splits):
                    #print split
                    tf.image_summary(layer[0]+'_%d'%i,split)
            if name == 'joinr':
                net[rank] = _join_layer(net[rank+1],net[rank])

    return net[1]

def _block_layer(net, nb_filter, filter_size, out = False):
    # net = tflearn.conv_2d(net, nb_filter = nb_filter, filter_size = filter_size, activation = 'relu')
    # net = tflearn.layers.normalization.batch_normalization(net)
    net = tflearn.conv_2d(net, nb_filter=nb_filter, filter_size=filter_size)
    net = tflearn.layers.batch_normalization(net)
    if out == False:
        net = tflearn.activations.leaky_relu(net)
    else:
        net = tflearn.activations.tanh(net)
    return net


def _join_layer(upnet, downnet):
    upnet = tflearn.upsample_2d(upnet, 2)
    upnet = tflearn.layers.normalization.batch_normalization(upnet)
    downnet = tflearn.layers.normalization.batch_normalization(downnet)
    downnet = tflearn.merge([upnet, downnet], 'concat', axis=3)
    return downnet

if __name__ == '__main__':
    X = tf.placeholder('float',[1, 256, 256, 6])
    network = net(X)
    with tf.Session() as sess:
        writer = tf.train.SummaryWriter('./logs/nn_logs',sess.graph)
        merged = tf.merge_all_summaries()
        tf.initialize_all_tables().run()
