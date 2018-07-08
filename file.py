from __future__ import division, print_function, absolute_import
import tflearn
from tflearn.data_utils import to_categorical, pad_sequences

import csv
import numpy as np
from sklearn import metrics, cross_validation

import tensorflow as tf

import random
import time
import itertools
import string

import getopt
import sys
import os
#The learning rate the of RNN model
LR=0.001
MODEL_NAME = 'emotion-{}-{}.model'.format(LR,'4conv-basic-video')
#Making a list of the labels that we require 
labels =["happy","sad","surprise","disgust"]
i=0
x=[]
y=[]
#Importing the data in the text format from the .txt file 
with open("/home/pratyush/Desktop/paraprocess/happy.txt", "r") as file:
	lines = file.read().split('\n\n')
#Appending the data in a list to create a data set
x.append(lines)
for j in range(len(lines)):
	y.append(labels[i])#appending the label for the data simultaneously on a list y
i+=1
with open("/home/pratyush/Desktop/paraprocess/sad.txt", "r") as file:
	lines = file.read().split('\n\n')
x.append(lines)
for j in range(len(lines)):
	y.append(labels[i])
i+=1
with open("/home/pratyush/Desktop/paraprocess/surprise.txt", "r") as file:
	lines = file.read().split('\n\n')
x.append(lines)
for j in range(len(lines)):
	y.append(labels[i])
i+=1
with open("/home/pratyush/Desktop/paraprocess/disgust.txt", "r") as file:
	lines = file.read().split('\n\n')
x.append(lines)
for j in range(len(lines)):
	y.append(labels[i])	
i+=1
x=list(itertools.chain(*x))#used to conevert the list of lists to a single list of data
print (x)
print (y)
for i in range(len(y)):
    y[i] = labels.index(y[i])
print (y)

  
model_size = 110
nb_epochs = 50   

MAX_FILE_ID = 50000
# x=np.array(x)
# y=np.array(y)

X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(x, y,test_size=0.15, random_state=2017)

Y_train = to_categorical (Y_train, nb_classes = 4)
Y_test = to_categorical (Y_test, nb_classes = 4)

### Process vocabulary

print('Process vocabulary')

vocab_processor = tflearn.data_utils.VocabularyProcessor(max_document_length = model_size, min_frequency = 0)
X_train = np.array(list(vocab_processor.fit_transform(X_train)))
X_test = np.array(list(vocab_processor.fit_transform(X_test)))

X_train = pad_sequences(X_train, maxlen=model_size, value=0.)
X_test = pad_sequences(X_test, maxlen=model_size, value=0.)

n_words = len(vocab_processor.vocabulary_)
print('Total words: %d' % n_words)

### Models

print('Build model')

net = tflearn.input_data([None, model_size])
net = tflearn.embedding(net, input_dim=n_words, output_dim=128)
net = tflearn.lstm(net, 128, dropout=0.5)
net = tflearn.fully_connected(net, n_units = len(labels) , activation='softmax')
net = tflearn.regression(net, optimizer='adam', learning_rate=0.001,
                         loss='categorical_crossentropy')

print ('Train model')

model = tflearn.DNN(net, tensorboard_verbose=3,tensorboard_dir='log')
if os.path.exists('{}.meta'.format(MODEL_NAME)):
	model.load(MODEL_NAME)
	print('model loaded!')

print ('Predict')
model.fit(X_train, Y_train, n_epoch=150,validation_set=(X_test, Y_test), show_metric=True,
          batch_size=34)
print ('Testing sample')
#The location of the txt file of user defined test data
with open("/home/pratyush/Desktop/paraprocess/stry.txt", "r") as file:
	test = file.read().split('\n\n')
	print(test)
vocab_processor = tflearn.data_utils.VocabularyProcessor(max_document_length = model_size, min_frequency = 0)
test = np.array(list(vocab_processor.fit_transform(test)))
test = pad_sequences(test, maxlen=model_size, value=0.)
model_out=model.predict(test)

if np.argmax(model_out) == 0:print('Happy')
if np.argmax(model_out) == 1:print('Sad')
if np.argmax(model_out) == 2:print('Surprise')
if np.argmax(model_out) == 3:print('Disgust')
