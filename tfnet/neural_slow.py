# Copyright (c) 2015-2016 Anish Athalye. Released under GPLv3.

import numpy as np
import tensorflow as tf

import vgg

CONTENT_LAYER = 'relu4_2'
STYLE_LAYERS = ('relu1_1', 'relu2_1', 'relu3_1', 'relu4_1')


def stylize(content, style, iterations=50,
            content_weight=5e0, style_weight=1e2, tv_weight=1e2,
            learning_rate=0.01, print_iterations=None,
            checkpoint_iterations=None):
    """
    Stylize images.

    This function yields tuples (iteration, image); `iteration` is None
    if this is the final image (the last iteration).  Other tuples are yielded
    every `checkpoint_iterations` iterations.

    :rtype: iterator[tuple[int|None,image]]
    """
    shape = (1,) + content.shape
    style_shape = (1,) + style.shape
    content_features = {}
    style_features = {}

    # compute content features in feedforward mode
    g = tf.Graph()
    with g.as_default(), g.device('/cpu:0'), tf.Session() as sess:
        image = tf.placeholder('float', shape=shape)
        net = vgg.net(image)
        content_pre = np.array([content - vgg.MEAN_PIXEL])
        content_features[CONTENT_LAYER] = net[CONTENT_LAYER].eval(
            feed_dict={image: content_pre})

        # compute style features in feedforward mode
        g = tf.Graph()
        with g.as_default(), g.device('/cpu:0'), tf.Session() as sess:
            image = tf.placeholder('float', shape=style_shape)
            net = vgg.net(image)
            style_pre = np.array([style - vgg.MEAN_PIXEL])
            for layer in STYLE_LAYERS:
                features = net[layer].eval(feed_dict={image: style_pre})
                features = np.reshape(features, (-1, features.shape[3]))
                gram = np.matmul(features.T, features) / features.size
                style_features[layer] = gram

    # make stylized image using backpropogation
    with tf.Graph().as_default():
        initial = tf.random_normal(shape) * 0.256
        image = tf.Variable(initial)
        net = vgg.net(image)

        # content loss
        content_loss = content_weight * (2 * tf.nn.l2_loss(
            net[CONTENT_LAYER] - content_features[CONTENT_LAYER]) /
                                         content_features[CONTENT_LAYER].size)
        # style loss
        style_loss = 0
        style_losses = []
        for style_layer in STYLE_LAYERS:
            layer = net[style_layer]
            _, height, width, number = map(lambda i: i.value,
                                           layer.get_shape())
            size = height * width * number
            feats = tf.reshape(layer, (-1, number))
            gram = tf.matmul(tf.transpose(feats), feats) / size
            style_gram = style_features[style_layer]
            style_losses.append(
                2 * tf.nn.l2_loss(gram - style_gram) / style_gram.size)
        style_loss = style_weight * reduce(tf.add, style_losses)
        # total variation denoising
        tv_y_size = _tensor_size(image[:, 1:, :, :])
        tv_x_size = _tensor_size(image[:, :, 1:, :])
        tv_loss = tv_weight * 2 * (
            (tf.nn.l2_loss(image[:, 1:, :, :] - image[:, :shape[1] - 1, :, :]) /
             tv_y_size) +
            (tf.nn.l2_loss(image[:, :, 1:, :] - image[:, :, :shape[2] - 1, :]) /
             tv_x_size))
        # overall loss
        loss = content_loss + style_loss + tv_loss

        # optimizer setup
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

        # optimization
        best_loss = float('inf')
        best = None
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            for i in range(iterations):
                _, this_loss = sess.run([train_step, loss])
                print('Iteration %d/%d' % (i + 1, iterations))

                if i % 20 == 0 or i == iterations - 1:
                    if this_loss < best_loss:
                        best_loss = this_loss
                        best = image.eval()
                if i == iterations - 1:
                    return best.reshape(shape[1:]) + vgg.MEAN_PIXEL


def _tensor_size(tensor):
    from operator import mul
    return reduce(mul, (d.value for d in tensor.get_shape()), 1)