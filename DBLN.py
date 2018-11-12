# -*- coding: utf-8 -*-
"""
Created on Sat Apr 29 05:02:18 2017

@author: nishanth
"""

from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from sklearn import datasets
from sklearn.datasets import load_iris
from nolearn.dbn import DBN
import numpy as np
import pylab as pl
import time
digits = datasets.load_digits()
data = digits.images.reshape((digits.images.shape[0], -1))
(trainX, testX, trainY, testY) = train_test_split(
	data/255 , digits.target.astype("int0"), test_size = 0.33)

dbn = DBN(
	[trainX.shape[1], 300, 10],
	learn_rates = 0.3,
	learn_rate_decays = 0.9,
	epochs = 10,
	verbose = 1)
dbn.fit(trainX, trainY)
preds = dbn.predict(testX)
print classification_report(testY, preds)
for i in np.random.choice(np.arange(0, len(testY)), size = (10,)):
	# classify the digit
     print len(testX[i])
     
     pred = dbn.predict(np.atleast_2d(testX[i]))
     image = (testX[i] * 255).reshape((8, 8))
     print "Actual digit is {0}, predicted {1}".format(testY[i], pred[0])
     pl.imshow(image,cmap = "gray")
     pl.show()
     