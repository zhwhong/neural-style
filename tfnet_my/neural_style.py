# Copyright (c) 2015-2016 Anish Athalye. Released under GPLv3.

import argparse
import os

from skimage import io
# from deal_image import *
from stylize_new import stylize
# default arguments
# ITERATIONS = 10
VGG_PATH = 'imagenet-vgg-verydeep-19.mat'


def build_parser():
    parser = argparse.ArgumentParser(description = 'neural style')
    # parser.add_argument('--styles',
    #                    nargs='+', help='one or more style images',
    #                    metavar='STYLE', default='./styles/wave.jpg')
    # parser.add_argument('--contents_dir', help='content image directory',
    #                    metavar='CONTENT', default='./data/train/people/')
    # parser.add_argument('--tests_dir', help='test image directory',
    #                    metavar='TEST', default='./data/test/my-test/')

    parser.add_argument('-s', '--style-image', type = str, default = 'starry_night.jpg',
                        help = 'the style image')
    parser.add_argument('-cd', '--content-dir', type = str, default = 'my-train',
                        help = 'the content image directory')
    parser.add_argument('-td', '--test-dir', type = str, default = 'my-test',
                        help = 'the test image directory')
    parser.add_argument('-i', '--iterations', type = int, default = 5,
                        help = 'the train iterations')
    parser.add_argument('--network',
                        help='path to network parameters (default %(default)s)',
                        metavar='VGG_PATH', default=VGG_PATH)
    return parser

def main():
    parser = build_parser()
    options = parser.parse_args()
    network = VGG_PATH
    if not os.path.isfile(VGG_PATH):
        parser.error("Network %s does not exist. (Did you forget to download it?)" % options.network)
    ckpt_dir = './ckpt-dir'
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    stylepath = './styles/' + options.style_image
    cd = './data/train/' + options.content_dir + '/'
    td = './data/test/' + options.test_dir + '/'

    style_image = io.imread(stylepath)

    stylize(
        network = network,
        style = style_image,
        iterations = options.iterations,
        contents_dir = cd,
        tests_dir = td,
        ckpt_dir = ckpt_dir
    )

if __name__ == '__main__':
    main()
