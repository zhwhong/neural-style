import tensorflow as tf
import tflearn


def _conv(network, filter, filter_size, strides=1):
    network = tflearn.conv_2d(network, nb_filter=filter,
                              filter_size=filter_size, strides=strides)
    network = tf.nn.batch_normalization(network, 0, 1.0, 0.0, 1.0, 1e-5)
    network = tflearn.activations.leaky_relu(network)
    return network


def _deconv(network, filter, filter_size, output_shape, out=False):
    network = tflearn.conv_2d_transpose(network, nb_filter=filter,
                                        filter_size=filter_size,
                                        output_shape=output_shape)
    # network = \
    #     tf.image.resize_image_with_crop_or_pad(image=network,
    #                                            target_height=output_shape[0],
    #                                            target_width=output_shape[1])
    if not out:
        network = tflearn.activations.leaky_relu(network)
    return network


def net(input_image):
    _, height, width, channel = map(lambda i: i.value, input_image.get_shape())
    conv1 = _conv(input_image, 64, 5, 2)
    conv1_1 = _conv(conv1, 48, 3)
    conv2 = _conv(conv1_1, 128, 5, 2)
    conv2_1 = _conv(conv2, 96, 3)
    conv3 = _conv(conv2_1, 256, 5, 2)
    conv3_1 = _conv(conv3, 192, 3)
    deconv1 = _deconv(conv3_1, 128, height / 4, [height, width]) + conv2
    conv4_1 = _conv(deconv1, 160, 3)
    deconv2 = _deconv(conv4_1, 64, height / 2, [height, width]) + conv1
    conv5_1 = _conv(deconv2, 96, 3)
    deconv3 = _deconv(conv5_1, 3, 8, [height, width], out=True)
    return deconv3