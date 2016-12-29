# Copyright (c) 2015-2016 Anish Athalye. Released under GPLv3.

import glob
import os
from sys import stderr

import tensorflow as tf
#from skimage import io

import pgg2
import vgg
from deal_image import *

_CONTENT_LAYER = 'relu4_2'
_STYLE_LAYERS = ('relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1')

ckpt_dir = './ckpt'

try:
    reduce
except NameError:
    from functools import reduce


def get_style_features(style, network):
    style_features = {}
    style_shape = (1,) + style.shape
    # compute style features in feedforward mode
    g = tf.Graph()
    with g.as_default(), g.device('/cpu:0'), tf.Session() as sess:
        image = tf.placeholder('float', shape=style_shape)
        print "style-shape", image
        net, mean_pixel = vgg.net(network, image)
        style_pre = np.array([vgg.preprocess(style, mean_pixel)])
        for layer in _STYLE_LAYERS:
            features = net[layer].eval(feed_dict={image: style_pre})
            features = np.reshape(features, (-1, features.shape[3]))
            gram = np.matmul(features.T, features) / features.size
            # gram shape: [64, 64], [128, 128], [256, 256], [512, 512], [512,
            #  512]
            style_features[layer] = gram
    return style_features


def stylize(network, style, iterations, contents_dir, tests_dir,ckpt_dir=None):
    # VGG network
    content_shape = (1,) + (256, 256, 3)  # -> shape = [1, 256, 256, 3]
    style_features = get_style_features(style, network)

    content_image = tf.placeholder('float', shape=content_shape)
    content_net, mean_pixel = vgg.net(network, content_image)
    YZ = tf.placeholder('float', shape=(1, 256, 256, 6))
    image = pgg2.net(YZ)  # content tensor, shape=[?, 256, 256, 3]
    tf.image_summary('before',image)
    W = tf.constant(128.0, shape = [1,256,256,3])
    b = tf.constant(128.0, shape = [1,256,256,3])
    image2 = tf.add(tf.mul(image, W), b)
    tf.image_summary('after',image2)

    net2, _ = vgg.net(network, vgg.preprocess(image2,mean_pixel))

    global_step = tf.Variable(0, name='global_step', trainable=False)
    saver = tf.train.Saver()

    print 'into the function stylize'

    def print_progress(i, j, last=False):
        stderr.write('Iteration %d/%d----- Image %d/%d\n' % (
            j + 1, iterations, i + 1, len(glob.glob(contents_dir + '/*'))))

    filelist = os.listdir(contents_dir)
    num_files = len(filelist)
    filetest = os.listdir(tests_dir)
    num_tests = len(filetest)

    content_feature = content_net[_CONTENT_LAYER]
    _, height, width, number = map(lambda i: i.value, content_feature.get_shape())
    content_size = height * width * number
    content_loss = 1 * (2 * tf.nn.l2_loss(
        net2[_CONTENT_LAYER] - content_feature) / content_size)
    tf.scalar_summary('content-loss', content_loss)

    style_loss = 0
    style_losses = []
    for style_layer in _STYLE_LAYERS:
        layer = net2[style_layer]
        _, height, width, number = map(lambda i: i.value, layer.get_shape())
        size = height * width * number
        feats = tf.reshape(layer, (-1, number))
        gram = tf.matmul(tf.transpose(feats), feats) / size
        style_gram = style_features[style_layer]
        style_losses.append(2 * tf.nn.l2_loss(gram - style_gram) /
                            style_gram.size)
    #style_loss1 = tf.reduce_sum(style_losses)
    style_loss += reduce(tf.add, style_losses)
    tf.scalar_summary('style-loss', style_loss)
    #print style_loss, style_loss1

    # total variation denoising
    tv_y_size = _tensor_size(image[:,1:,:,:])
    tv_x_size = _tensor_size(image[:,:,1:,:])
    tv_loss = 1e2 * 2 * (
            (tf.nn.l2_loss(image[:,1:,:,:] - image[:,:content_shape[1]-1,:,:]) /
                tv_y_size) +
            (tf.nn.l2_loss(image[:,:,1:,:] - image[:,:,:content_shape[2]-1,:]) /
                tv_x_size))
    # overall loss
    total_loss = content_loss + 1*style_loss + tv_loss
    tf.scalar_summary('total-loss',total_loss)


    # optimizer set
    train_step = tf.train.AdamOptimizer(0.01).minimize(total_loss)

    with tf.Session() as sess:
        writer = tf.train.SummaryWriter("./logs/nn_logs",sess.graph)
        merged = tf.merge_all_summaries()

        sess.run(tf.initialize_all_variables())

        ckpt = tf.train.get_checkpoint_state(ckpt_dir)
        if ckpt and ckpt.model_checkpoint_path:
            print ckpt.model_checkpoint_path
            saver.restore(sess, ckpt.model_checkpoint_path)
        start = global_step.eval()
        print("start from:", start)
        print contents_dir,tests_dir

        for epoch in range(start, start+iterations):
            #train
            for i in range(num_files):
                # last_step = ((epoch == iterations - 1) and (i == num_files - 1))
                # print_progress(i, epoch-start, last=last_step)
                print_progress(i, epoch-start)
                filename = os.path.join(contents_dir, filelist[i])
                content = imread(filename)
                content_pre = np.array([vgg.preprocess(content, mean_pixel)])

                y = np.reshape(content_pre, (1, 256, 256, 3))
                z = np.random.normal(size=[1, 256, 256, 3])
                yz = np.append(y, z, axis=3)
                #yz = np.zeros([1,256,256,6])
                #print "[PGG-IN]yz:",yz

                feed_dict = {
                    YZ: yz, content_image: content_pre
                }
                _,loss, images,imagess,summary = sess.run([train_step,total_loss, image,image2,merged],
                                           feed_dict=feed_dict)
                #print "epoch=%d files=%d[PGG-OUT]images:"%(epoch,i),images
                imgs = images.reshape(content_shape[1:])
                imsave('./output/image1_%d_%d.jpg'%(epoch,i),imgs)
                imgss = imagess.reshape(content_shape[1:])
                imsave('./output/image2_%d_%d.jpg'%(epoch,i), imgss)

                print "epoch=%d files=%d this_loss:%d"%(epoch,i,loss)
                writer.add_summary(summary,epoch*num_files+i)
                #if i % 20 == 0:
                    #test
            for j in range(num_tests):
                filename = os.path.join(tests_dir,filetest[j])
                fname,ftyle = os.path.splitext(os.path.basename(filename))
                test = imread(filename)
                test_pre = np.array([vgg.preprocess(test, mean_pixel)])

                y = np.reshape(test_pre, (1, 256, 256, 3))
                z = np.random.normal(size=[1, 256, 256, 3])
                yz = np.append(y, z, axis=3)
                images = sess.run(image2,feed_dict={YZ: yz})
                out = vgg.unprocess(images.reshape(content_shape[1:]),
                                mean_pixel)
                imsave('./output/%s_%d.jpg'%(fname, epoch), out)

            if epoch == start+iterations-1 :
                global_step.assign(epoch).eval()
                saver.save(sess, ckpt_dir + '/model.ckpt',global_step=global_step)

def _tensor_size(tensor):
    from operator import mul
    return reduce(mul, (d.value for d in tensor.get_shape()), 1)
