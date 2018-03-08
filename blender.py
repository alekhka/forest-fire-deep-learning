#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 01:37:52 2017

@author: alekhka
"""

from PIL import Image

import numpy
from skimage import color, io
import scipy
import random
import os
i=0
for filename in os.listdir('.'):
    
    ip=filename
    r=random.sample(xrange(1,4), 1)
    fire='fire'+str(r[0])+'.jpg'
    
    img = io.imread(fire)
    bkg = io.imread(filename)
    
    Y = img.astype('uint8')
    X = bkg.astype('uint8')

    i=i+1
    X = scipy.misc.imresize(X,(256,256,3))
    Y = scipy.misc.imresize(Y,(256,256,3))

    #print Y.shape,X.shape
    img = Image.fromarray(Y, 'RGB')
    bkg = Image.fromarray(X, 'RGB')
    
    op="op"+str(i)+".png"
    #print img.size
    #print bkg.size
    Image.blend(img,bkg,0.4).save(os.path.join('op3',op))