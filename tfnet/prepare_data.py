import glob
import random

ILSVRC_DIR = '/baina/sda1/data/ILSVRC/Data/CLS-LOC/train'

dirs = glob.glob(ILSVRC_DIR + '/*')
for d in dirs:
    files = glob.glob(d + '/*')
    for f in files:
        if random.random() < 0.1:
            print f