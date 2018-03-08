#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 09:31:11 2017

@author: alekhka
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Nov 09 18:34:52 2016

@author: Alekh
"""
# import png
import numpy
# from PIL import Image
# import os
from sklearn.neural_network import MLPClassifier
from skimage import color, io
import pickle
import scipy
import os
from Naked.toolshed.shell import execute_js, muterun_js
from gcm import GCM
#import sys
from PIL import Image, ImageChops
#import matplotlib.pyplot as plt
Sensitivity = 90
reg_ids='fCPEJb4yegQ:APA91bGf7svVqOqlGmawntpdprl9wJbKqbg_nML20GeMlC_5SCKB8T3lzKxVlL_5hsjUtDzMQ_OicQ8bNDTtwuBlY0yEcCJaKwrYEtkaBf6WOZy38ySuMwh2OrKZ4CCOHziLhpOJtOqX'
API_KEY='AAAAYMraJdk:APA91bEVBFy1S4pY-97eSGMxIL_mD4slU-H1k-oDlXPWDd7uI9t4qRuuCrC66nIGbVW4jkFUFsFNa40NNShzeiZ3T5W7LqiLhnEI2E4cnCAP_T3XhOLn7J9nFpw9wy4eiQyBydFv5EHs'

def sendNotif():
    gcm = GCM(API_KEY)
    
    data = {'icon':'fire','param1': '0.600015', 'param2': '24.139391'}
    '''
    # Downstream message using JSON request
    reg_ids = ['token1', 'token2', 'token3']
    response = gcm.json_request(registration_ids=reg_ids, data=data)
    
    # Downstream message using JSON request with extra arguments
    res = gcm.json_request(
        registration_ids=reg_ids, data=data,
        collapse_key='uptoyou', delay_while_idle=True, time_to_live=3600)'''

    # Topic Messaging
    success = execute_js('/home/alekhka/Desktop/NasaFire/server.js','12.07 77.71')
    print success

k = 0
j = 0
i = 4000


percent = 0.0000;

limit = 6
# Loop through all provided arguments
net = pickle.load(open("model2.sav", 'rb'))
while 1:
    for filename in os.listdir('.'):
        if (filename.endswith('.png')):
        
            # Attempt to open an image file
            # imageX = Image.open(filename)
            
            imageX = io.imread(filename)
            imageX = color.rgb2gray(imageX)
            #Image._show(imageY)
            if filename.startswith('i'):
                e=0
            else:
                e=1
            
            j = j + 1
            # print "loaded imgs"
            imageX*=255
            
            X = imageX.astype('uint8')
                
            '''im = Image.fromarray(Ynew,'L')
            im.save("zzzzzz"+filename)'''
            
            '''for i in range(0,240):
                for l in range(0,320):
                    imageY/=65535'''
                     
            #X = numpy.asarray(imageX, dtype='uint8')
            #Y = numpy.asarray(Y, dtype='uint8')
            X = scipy.misc.imresize(X,(60,80))
            
            
            #print "X shape"
            #print X.shape
            #del imageX, imageY
            #X = numpy.ravel(X)
            #Y = numpy.ravel(Y)
            X = X.reshape(-1, 4800)
            
        
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
        
            #percent = float(j * 100 / limit)
            #sys.stdout.write("\r %i" % percent)
            # sys.stdout.write("-")
            #sys.stdout.flush()
            #print str(percent) + "% "
            # print "score:"
            # print S
            i = i + 4
            print j%27
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
            
            #Y = numpy.asarray(Y, dtype='uint8')
            '''Y = numpy.reshape(Y, (60, 80))
            im = Image.fromarray(Y, 'L')
            im.save("zzySK2" + filename)'''
            
            Yp = net.predict(X)
            
            #print Yp
            if e==1:
                sendNotif()
                print 'FIRE DETECTED'
                while 1:
                    vure=5
                break
            
            
            
X = numpy.reshape(X,(60,80))
im = Image.fromarray(X,'L')
im.show()    


#filename3 = 'zzzzzsk_modelgood1.sav'
#pickle.dump(net, open(filename3, 'wb'))

#loaded_model = pickle.load(open("finalized_model.sav", 'rb'))