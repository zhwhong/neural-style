import numpy
from PIL import Image, ImageEnhance


def filter_image(image, contrast=1.5, color=1.5):
    img = Image.fromarray(numpy.uint8(image))
    img = contrast_image(img, contrast)
    img = color_image(img, color)
    return numpy.array(img)


def filter_image_file(image_file):
    img = Image.open(image_file)
    return filter(img)


def color_image(img, weight):
    color = ImageEnhance.Color(img)
    color_img = color.enhance(weight)
    return color_img


def contrast_image(img, weight):
    contrast = ImageEnhance.Contrast(img)
    contrast_img = contrast.enhance(weight)
    return contrast_img


def brightness_image(img, weight):
    brightness = ImageEnhance.Brightness(img)
    bright_img = brightness.enhance(weight)
    return bright_img


if __name__ == '__main__':
    image = filter('test.jpg')
    image.save('./output.jpg')
