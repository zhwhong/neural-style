# Copyright (c) 2015-2016 Anish Athalye. Released under GPLv3.

import argparse
import os

from stylize_new import stylize, crop_image, imread

ROOT = os.path.dirname(__file__)
DEFAULT_STYLE = os.path.join(ROOT, 'data/style.jpg')
DEFAULT_CONTENT = os.path.join(ROOT, 'data/content/')
DEFAULT_TEST = os.path.join(ROOT, 'data/test/')

dshape = [512, 512, 3]

def build_parser():
    parser = argparse.ArgumentParser(description='neural style')
    parser.add_argument('-s', '--style-image', type=str, default=DEFAULT_STYLE,
                        help='the style image')
    parser.add_argument('-cd', '--content-dir', type=str,
                        default=DEFAULT_CONTENT,
                        help='the content image directory')
    parser.add_argument('-td', '--test-dir', type=str, default=DEFAULT_TEST,
                        help='the test image directory')
    parser.add_argument('-i', '--iterations', type=int, default=5,
                        help='the train iterations')
    parser.add_argument('-o', '--output', default='output',
                        help='output directory')
    parser.add_argument('-d', '--device', default='/gpu:0',
                        help='running device')
    return parser


def main():
    parser = build_parser()
    options = parser.parse_args()
    ckpt_dir = './ckpt-dir'
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    style_image = imread(options.style_image)
    name = os.path.splitext(os.path.basename(options.style_image))[0]
    style_image = crop_image(style_image, dshape)

    stylize(
        style=style_image,
        dshape=dshape,
        iterations=options.iterations,
        contents_dir=options.content_dir,
        tests_dir=options.test_dir,
        ckpt_dir=ckpt_dir,
        output=options.output,
        device=options.device
    )


if __name__ == '__main__':
    main()
