#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 08:25:13 2017

@author: alekhka
"""


# import png
import numpy
# from PIL import Image
import os
from sklearn.neural_network import MLPClassifier
from skimage import color, io
import pickle
import scipy
import random
#import sys
from PIL import Image
#import matplotlib.pyplot as plt
k = 0
j = 0
i = 1

f=random.sample(xrange(1,3401),3400)
f=numpy.array(f, dtype=int)
percent = 0.0000;

limit = 1000;
# Loop through all provided arguments
net = MLPClassifier(hidden_layer_sizes=(2000, 200), activation='relu', solver='adam', alpha=0.1, batch_size='auto',
                   learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=200, shuffle=True,
                   random_state=None, tol=0.0001, verbose=False, warm_start=False, momentum=0.9,
                   nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999,
                   epsilon=1e-08)

for j in range(1, limit):
    r=random.sample(xrange(0,2),1)
    n=f[i]%1700
    if r[0]==0:
        try:
            imageX = io.imread(os.path.join('data/train/imgs',"img"+str(n)+".png"))
            Y=[0]
        except Exception,e:
            print e
            pass
    else:
        try:
            imageX = io.imread(os.path.join('data/train/op3',"op"+str(n)+".png"))
            Y=[1]
        except Exception,e:
            print e
            pass
        
            
    print n
    imageX = color.rgb2gray(imageX)*255
    #Image._show(imageY)
    
    
    j = j + 1
    # print "loaded imgs"
    #imageY*=255
    
    X = imageX.astype('uint8')
        
    '''im = Image.fromarray(Ynew,'L')
    im.save("zzzzzz"+filename)'''
    
    '''for i in range(0,240):
        for l in range(0,320):
            imageY/=65535'''
             
    #X = numpy.asarray(imageX, dtype='uint8')
    #Y = numpy.asarray(Y, dtype='uint8')
    X = scipy.misc.imresize(X,(60,80))
    #Y = scipy.misc.imresize(Y,(60,80))
    
    #print "X shape"
    #print X.shape
    #del imageX, imageY
    #X = numpy.ravel(X)
    #Y = numpy.ravel(Y)
    X = X.reshape(-1, 4800)
    #Y = Y.reshape(-1, 4800)

    '''if i == 1:
        Xf = X
        Yf = Y
    else:
        Xf = numpy.vstack((Xf, X))
        Yf = numpy.vstack((Yf, Y))'''

    # width, height = X.size
    # width2, height2 = Y.size

    # print width, height
    # print width2, height2


    # Yp = net.predict(Xs)
    # S = numpy.sum((Yp[1:100] - Ys[1:100]) / 100)

    percent = float(j * 100 / limit)
    #sys.stdout.write("\r %i" % percent)
    # sys.stdout.write("-")
    #sys.stdout.flush()
    print str(percent) + "% "
    # print "score:"
    # print S
    i = i + 1
    # result = numpy.asarray(Yp, dtype='uint8')
    # print result

    '''for i in range(0,240):
        for l in range(0,320):
            if Yb[i,l]==0:
                k=k+1'''
    # else:
    # print result[i]
    # result1 = result.reshape(240,320,3)
    # result2 = result1.reshape(240,-1)
    '''print " k " + str(k)'''
    # filename3 = "zz" + filename2
    # print result1
    # im = png.from_array(result1,'RGB')
    # im.save(filename3)
    #print Xf.shape
    #print Yf.shape
    #print "Loaded " + filename
    print "fitting"
    C=[0,1]
    net.partial_fit(X, Y, C)
    print Y
    print "fitted"
    print "\n"
    print j
    #Y = numpy.asarray(Y, dtype='uint8')
    '''Y = numpy.reshape(Y, (60, 80))
    im = Image.fromarray(Y, 'L')
    
    im.save("zzySK2" + filename)'''
    
    if j==limit:
        print "almost"
    
    

X = numpy.reshape(X, (60, 80))
im = Image.fromarray(X, 'L')
print Y
im.save("zzySK3labelsample" + '.png')
filename3 = 'model1.sav'
pickle.dump(net, open(filename3, 'wb'))

