import tensorflow as tf
import tflearn


def block(input, nb_filter, filter_size):
    network = tflearn.conv_2d(input, nb_filter=nb_filter,
                              filter_size=filter_size)
    network = tf.nn.batch_normalization(network, 0, 1.0, 0.0, 1.0, 1e-5)
    network = tflearn.activations.leaky_relu(network)
    return network


def net(input_image):
    # _, height, width, channel = map(lambda i: i.value, input_image.get_shape())
    conv1 = block(input_image, 64, 5)
    conv2 = block(conv1, 32, 5)
    conv3 = block(conv2, 64, 3)
    conv4 = block(conv3, 32, 5)
    conv5 = block(conv4, 48, 5)
    conv6 = block(conv5, 32, 5)
    out = tflearn.conv_2d(conv6, nb_filter=3, filter_size=3, bias=False)
    return out