# EVT方法 W-SVM 可以直接用
'''
Analyzing the prior work and their distribution based on Extreme Value Theory (EVT)
The distributions follow Weibull distribution
'''
import os, sys, pickle, glob
import os.path as path
import argparse
import scipy.spatial.distance as spd
import scipy as sp
from scipy.io import loadmat
import numpy as np
from openmax_utils import *

try:
    import libmr
except ImportError:
    print ("ERROR : Error importing LibMR library... ")
    print ("Please install the libmr library using : cd libMR/; ./compile.sh")
    sys.exit()

NCHANNELS = 1

# The default eucos distance_Type is euclidean distance with a combination of Cosine distance.
def weibull_distribution_fitting(meanFilesPath, distanceFilesPath, tailsize = 10, distance_type = 'eucos'):
                        
    """
    Fit the weibull distribution model for each category by reading the data through distance files, mean activation vector.

    Input:
    *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
    meanFilesPath : contains path to files with pre-computed mean-activation vector
    distanceFilesPath : contains path to files with pre-computed distances for images from MAV

    Returns:
    *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
    weibull_distribution_model : Perform Extreme Value Theory (EVT) based analysis using tails of distances 
                and save weibull model parameters for re-adjusting the scores of softmax algorithm later in the model.
    """
    
    weibull_model = {}
    # for each category, read meanfile, distance file, and perform weibull fitting
    distance_scores = np.array(distanceFilesPath[distance_type])
    meantrain_vec = np.array(meanFilesPath)

    weibull_model['distances_%s'%distance_type] = distance_scores
    weibull_model['mean_vec'] = meantrain_vec
    weibull_model['weibull_model'] = []
    mr = libmr.MR()
    tailtofit = sorted(distance_scores)[-tailsize:]
    mr.fit_high(tailtofit, len(tailtofit))
    weibull_model['weibull_model'] += [mr]

    return weibull_model


# The default eucos distance_Type is euclidean distance with a combination of Cosine distance
def query_weibull_distribution(categoryname, weibullDistributionModel, distance_type = 'eucos'):
    """ Query through dictionary for Weibull model.
    Return in the order: [mean_vec, distances, weibull_model]
    
    Input:
    ------------------------------
    categoryname : name of ImageNet category in WNET format. E.g. n01440764
    weibull_model: dictonary of weibull models for 
    """
    
    category_weibull = []
    category_weibull += [weibullDistributionModel[categoryname]['mean_vec']]
    category_weibull += [weibullDistributionModel[categoryname]['distances_%s' %distance_type]]
    category_weibull += [weibullDistributionModel[categoryname]['weibull_model']]

    return category_weibull    

