# Copyright (c) 2015-2016 Anish Athalye. Released under GPLv3.

import os
import random
from sys import stderr

import tensorflow as tf

from image_utils import *
from tfnet.gens import generator

_CONTENT_LAYER = 'relu4_2'
_STYLE_LAYERS = ('relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1')

content_shape = (1,) + (512, 512, 3)  # -> shape = [1, 512, 512, 3]

content_weight = 1
style_weight = 10
tv_weight = 1e-2


def _tensor_size(tensor):
    from operator import mul
    return reduce(mul, (d.value for d in tensor.get_shape()), 1)


def get_style_features(style):
    style_features = {}
    style_shape = (1,) + style.shape
    # compute style features in feedforward mode
    g = tf.Graph()
    with g.as_default(), g.device('/cpu'), tf.Session(graph=g):
        image = tf.placeholder('float', shape=style_shape)
        print "style-shape", image
        net = vgg.net(image)
        style_pre = np.array([style - vgg.MEAN_PIXEL])
        for layer in _STYLE_LAYERS:
            features = net[layer].eval(feed_dict={image: style_pre})
            features = np.reshape(features, (-1, features.shape[3]))
            gram = np.matmul(features.T, features) / features.size
            style_features[layer] = gram
    # gram shape: [64, 64], [128, 128], [512, 512], [512, 512], [512,512]
    return style_features


def _content_loss(content_net, net2):
    content_feature = content_net[_CONTENT_LAYER]
    _, height, width, number = map(lambda i: i.value,
                                   content_feature.get_shape())
    content_size = height * width * number
    loss = 1 * (
        2 * tf.nn.l2_loss(
            net2[_CONTENT_LAYER] - content_feature) / content_size)
    return loss


def _style_loss(net2, style_features):
    loss = 0.
    style_losses = []
    for style_layer in _STYLE_LAYERS:
        layer = net2[style_layer]
        _, height, width, number = map(lambda i: i.value, layer.get_shape())
        size = height * width * number
        feats = tf.reshape(layer, (-1, number))
        gram = tf.matmul(tf.transpose(feats), feats) / size
        style_gram = style_features[style_layer]
        style_losses.append(
            2 * tf.nn.l2_loss(gram - style_gram) / style_gram.size)
    loss += reduce(tf.add, style_losses)
    return loss


def _tv_loss(image):
    # total variation denoising
    tv_y_size = _tensor_size(image[:, 1:, :, :])
    tv_x_size = _tensor_size(image[:, :, 1:, :])
    loss = 1e2 * 2 * (
        (tf.nn.l2_loss(
            image[:, 1:, :, :] - image[:, :content_shape[1] - 1, :, :]) /
         tv_y_size) +
        (tf.nn.l2_loss(
            image[:, :, 1:, :] - image[:, :, :content_shape[2] - 1, :]) /
         tv_x_size))
    return loss


def _total_loss(style_loss, content_loss, tv_loss):
    loss = content_weight * content_loss + style_weight * style_loss \
           + tv_weight * tv_loss
    return loss


def print_pixels(image, name='image'):
    for i in range(40, 200, 40):
        print '%s[%d, %d]: [%.2f, %.2f, %.2f]' % (name, i, i,
                                                  image[i][i][0],
                                                  image[i][i][1],
                                                  image[i][i][2])


def print_progress(n_pic, n_iter, iterations, files):
    stderr.write('Iteration %d/%d ----- Image %d/%d\n' % (
        n_iter + 1, iterations, n_pic + 1, files))


def stylize(style, dshape, iterations, contents_dir, tests_dir, ckpt_dir=None,
            output=None, device='/cpu'):
    print 'into the function stylize()'
    style_features = get_style_features(style)

    with tf.device(device):
        content_image = tf.placeholder('float', shape=content_shape)
        content_net = vgg.net(content_image)
        YZ = tf.placeholder('float', shape=(1, 512, 512, 6))
        image = generator.net(YZ)
        net2 = vgg.net(image)

        content_loss = _content_loss(content_net, net2)
        style_loss = _style_loss(net2, style_features)
        tv_loss = _tv_loss(image)

        # overall loss
        total_loss = _total_loss(style_loss, content_loss, tv_loss)

        # optimizer set
        train_step = tf.train.AdamOptimizer().minimize(total_loss)

    tf.scalar_summary('content-loss', content_loss)
    tf.scalar_summary('style-loss', style_loss)
    tf.scalar_summary('tv-loss', tv_loss)
    tf.scalar_summary('total-loss', total_loss)

    global_step = tf.Variable(0, name='global_step', trainable=False)
    saver = tf.train.Saver()

    with tf.Session() as sess:
        writer = tf.train.SummaryWriter("./logs/nn_logs", sess.graph)
        merged = tf.merge_all_summaries()

        sess.run(tf.initialize_all_variables())

        ckpt = tf.train.get_checkpoint_state(ckpt_dir)
        if ckpt and ckpt.model_checkpoint_path:
            # print ckpt.model_checkpoint_path
            saver.restore(sess, ckpt.model_checkpoint_path)

        start = global_step.eval()
        # start = 0
        print("start epoch:", start)
        print contents_dir, tests_dir

        filelist = [f for f in os.listdir(contents_dir)
                    if os.path.splitext(f)[1][1:] == 'jpg']
        num_files = len(filelist)
        filetest = [f for f in os.listdir(tests_dir)
                    if os.path.splitext(f)[1][1:] == 'jpg']
        num_tests = len(filetest)
        # iterations
        for epoch in range(start, start + iterations):
            random.shuffle(filelist)
            # train
            for i in range(num_files):
                print_progress(i, epoch - start, iterations, num_files)
                filename = os.path.join(contents_dir, filelist[i])
                content = imread(filename)
                content = crop_image(content, dshape)
                content_pre = np.array([content - vgg.MEAN_PIXEL])

                y = np.reshape(content_pre, (1, 512, 512, 3))
                z = np.random.normal(size=[1, 512, 512, 3])
                yz = np.append(y, z, axis=3)
                feed_dict = {YZ: yz, content_image: content_pre}

                _, loss, image_out, summary = sess.run(
                    [train_step, total_loss, image, merged],
                    feed_dict=feed_dict)

                """
                if epoch == start+iterations-1:
                    dic = {}
                    for v in tflearn.variables.get_all_trainable_variable():
                        dic[v.name] = v.eval()
                    print dic
                """

                print "epoch=%d files=%d this_loss:%f" % (epoch, i, loss)
                writer.add_summary(summary, epoch * num_files + i)

                out = image_out.reshape(content_shape[1:])
                out = out + vgg.MEAN_PIXEL
                print_pixels(out, 'image')

                if epoch % 5 == 0:
                    img = image_out.reshape(content_shape[1:]) + vgg.MEAN_PIXEL
                    print_pixels(img, 'pgg')
                    imsave('./%s/pgg_train_%d_%d.jpg' % (output, epoch, i), img)

            # test
            for j in range(num_tests):
                filename = os.path.join(tests_dir, filetest[j])
                fname, _ = os.path.splitext(os.path.basename(filename))
                test = imread(filename)
                test = crop_image(test, dshape)
                test_pre = np.array([test - vgg.MEAN_PIXEL])

                y = np.reshape(test_pre, (1, 512, 512, 3))
                z = np.random.normal(size=[1, 512, 512, 3])
                yz = np.append(y, z, axis=3)
                image_out = sess.run(image, feed_dict={YZ: yz})
                if epoch % 5 == 0:
                    out = image_out.reshape(content_shape[1:]) + vgg.MEAN_PIXEL
                    # print 'test-out: [%f,%f,%f] -> [%f,%f,%f]' % (
                    #     out[120][120][0] - 123.68, out[120][120][1] - 116.779,
                    #     out[120][120][2] - 103.939,
                    #     out[120][120][0], out[120][120][1], out[120][120][2])
                    imsave('./%s/%s_%d.jpg' % (output, fname, epoch), out)

            # save the checkpoint
            if epoch == start + iterations - 1:
                global_step.assign(epoch).eval()
                saver.save(sess, ckpt_dir + '/model.ckpt',
                           global_step=global_step)