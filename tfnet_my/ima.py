import os
import glob
import argparse
import numpy as np
from PIL import Image

HEIGHT = 256
WIDTH = 256
img_path = './data/test/my-test/'


def cut(image):

	img = Image.open(image)
	X, Y = img.size

	if X == WIDTH and Y == HEIGHT:
		print 'fit'
		return img
	if min(X,Y)<256:
		os.remove(image)
		return None
	#resize
	if X < Y:
		newWidth = WIDTH
		newHeight = float(HEIGHT) / X * Y
		#newHeight = int(newHeight)
	else:
		newHeight = HEIGHT
		newWidth = float(WIDTH) / Y * X
		#newWidth = int(newHeight)
	img.thumbnail((newWidth,newHeight),Image.ANTIALIAS)
	#img.resize((newWidth,newHeight),Image.ANTIALIAS)

	if img.size == (WIDTH,HEIGHT):
		print 'square',img.size
		return img

	#cut
	if newWidth < newHeight:
		region = (0, newHeight/2- HEIGHT/2 , WIDTH, newHeight/2 + HEIGHT/2)
	else:
		region = (newWidth/2 - WIDTH/2, 0, newWidth/2 + WIDTH/2, HEIGHT)
	img = img.crop(region)
	print img.size
	return img


if __name__ == '__main__':

	for i,files in enumerate(glob.glob(img_path+'*')):
		filepath,filename = os.path.split(files)
		#filtername,exts = os.path.splitext(filename)

		opfile = img_path
		if os.path.isdir(opfile) == False:
			os.mkdir(opfile)
		try:
			img = cut(files)
			if img is not None:
				img.save(files)
		except:
			print "loss 1 pic"
			os.remove(files)
			pass
    

