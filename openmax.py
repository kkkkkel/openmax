from __future__ import print_function
import tensorflow as tf
import keras
#from keras.datasets import cifar10
from keras.models import Sequential,load_model,save_model,model_from_config,model_from_json
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from tensorflow.keras.utils import to_categorical

from EVT_Weibull_Fitting import weibull_distribution_fitting, query_weibull_distribution
from compute_openmax import computeOpenMaxProbability, recalibrate_scores
from openmax_utils import compute_distance

import scipy.spatial.distance as spd
import h5py

import libmr

import numpy as np
import scipy

import pickle
import matplotlib.pyplot as plt

import ssl
import sys
import cv2
import glob

tf.compat.v1.disable_eager_execution()

ssl._create_default_https_context = ssl._create_unverified_context

#改为新闻的labels
#datasets
label=['0', ]

def seperate_data(x,y):
    ind = y.argsort()
    sort_x = x[ind[::-1]]
    sort_y = y[ind[::-1]]

    dataset_x = []
    dataset_y = []
    mark = 0

    for a in range(len(sort_y)-1):
        if sort_y[a] != sort_y[a+1]:
            dataset_x.append(np.array(sort_x[mark:a]))
            dataset_y.append(np.array(sort_y[mark:a]))
            mark = a
        if a == len(sort_y)-2:
            dataset_x.append(np.array(sort_x[mark:len(sort_y)]))
            dataset_y.append(np.array(sort_y[mark:len(sort_y)]))
    return dataset_x,dataset_y

#feature
def compute_feature(x,model):
    score = get_activations(model,14,x)
    fc8 = get_activations(model,13,x)
    return score,fc8

# center
def compute_mean_vector(feature):
    print("compute_mean_vector features = {}".format(feature.shape))
    mn = np.mean(feature, axis=0)
    print("compute_mean_vector mn = {}".format(mn.shape))
    return mn

def compute_distances(mean_feature,feature,category_name):
    eucos_dist, eu_dist, cos_dist = [], [], []
    eu_dist,cos_dist,eucos_dist = [], [], []
    print("mean_feature = {}".format(mean_feature))
    for feat in feature:
        eu_dist += [spd.euclidean(mean_feature, feat)]
        cos_dist += [spd.cosine(mean_feature, feat)]
        eucos_dist += [spd.euclidean(mean_feature, feat)/200. + spd.cosine(mean_feature, feat)]
    distances = {'eucos':eucos_dist,'cosine':cos_dist,'euclidean':eu_dist}
    return distances
                       
batch_size = 128
num_classes = 10
epochs = 50


#改为text的文本

# Delete
# input image dimensions
#data shape
img_rows, img_cols = 32, 32

# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 3, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 3, img_rows, img_cols)
    input_shape = (3, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 3)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 3)
    input_shape = (img_rows, img_cols, 3)


x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# convert class vectors to binary class matrices
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)


def get_activations(model, layer, X_batch):
    get_activations = K.function([model.layers[0].input,K.learning_phase()], [model.layers[layer].output])
    activations = get_activations([X_batch,0])[0]
    return activations

def get_correct_classified(pred,y):
    pred = (pred > 0.5 ) * 1
    res = np.all(pred == y,axis=1)
    return res

def create_model(model):

    output = model.layers[-1]
    
    # Combining the train and test set

    x_all = np.concatenate((x_train,x_test),axis=0)
    y_all = np.concatenate((y_train,y_test),axis=0)    
    pred = model.predict(x_all)
    index = get_correct_classified(pred,y_all)
    x1_test = x_all[index]
    y1_test = y_all[index]

    y1_test1 = y1_test.argmax(1)

    sep_x, sep_y = seperate_data(x1_test, y1_test1)

    feature = {}
    feature["score"] = []
    feature["fc8"] = []
    weibullDistributionModel = {}
    feature_mean = []
    feature_distance = []

    for i in range(len(sep_y)):
        print (i, sep_x[i].shape)
        weibullDistributionModel[label[i]] = {}
        score,fc8 = compute_feature(sep_x[i], model)
        mean = compute_mean_vector(fc8)
        distance = compute_distances(mean, fc8, sep_y)
        feature_mean.append(mean)
        feature_distance.append(distance)
    np.save('mean',feature_mean)
    np.save('distance',feature_distance)

def build_weibull(mean,distance,tail):
    weibullDistributionModel = {}    
    for i in range(len(mean)):
        weibullDistributionModel[label[i]] = {}        
        weibull = weibull_distribution_fitting(mean[i], distance[i], tailsize = tail)
        weibullDistributionModel[label[i]] = weibull
    return weibullDistributionModel
        
def compute_openmax(model,imagearr):
    mean = np.load('mean.npy')
    distance = np.load('distance.npy')


    alpharank_list = [20]  # 10改为20
    tail_list = [5]    
    total = 0
    for alpha in alpharank_list:
        weibullDistributionModel = {}
        openmax = None
        softmax = None        
        for tail in tail_list:
            weibullDistributionModel = build_weibull(mean, distance, tail) 
            openmax , softmax = recalibrate_scores(weibullDistributionModel, label, imagearr, alpharank=alpha)

    return np.argmax(softmax),np.argmax(openmax)

# tensor大小是不是应该改一下
def process_input(model,ind):
    imagearr = {}
    plt.imshow(np.squeeze(x_train[ind]))    
    plt.show()
    image = np.reshape(x_train[ind],(1,32,32,3))
    score5,fc85 = compute_feature(image, model)    
    imagearr['scores'] = score5
    imagearr['fc8'] = fc85
    return imagearr

def compute_activation(model,img):
    imagearr = {}
    print("compute_activation img = {}".format(img.shape))
    img = np.reshape(img,(1,32,32,3))
    score5,fc85 = compute_feature(img, model)
    imagearr['scores'] = score5
    imagearr['fc8'] = fc85
    return imagearr

def image_show(img,label):
    img = scipy.misc.imresize(np.squeeze(img),(32,32))
    img = img[:,0:32*32]    
    plt.imshow(np.squeeze(img), cmap='gray')
    print ('Character Label: ',np.argmax(label))
    plt.show()


def openmax_unknown_class(model):
    f = h5py.File('HWDB1.1subset.hdf5','r')
    total = 0
    i = np.random.randint(0,len(f['tst/y']))
    print ('label',np.argmax(f['tst/y'][i]))
    print (f['tst/x'][i].shape)
    imagearr = process_input(model,f['tst/x'][i])
    compute_openmax(model,imagearr)


def openmax_known_class(model,y):
    total = 0
    for i in range(15):
        j = np.random.randint(0,len(y_train[i]))
        imagearr = process_input(model,j)
        print (compute_openmax(model, imagearr))

