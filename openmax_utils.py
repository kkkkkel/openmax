# openmax最底层的结构
import os, sys, pickle, glob
import os.path as path
import argparse
import scipy.spatial.distance as spd
import scipy as sp
import numpy as np

# 获取文件的函数 注意参数是否需要修改
def parse_synsetfile(synsetfname):
    """ Read ImageNet 2012 file
    """
    categorylist = open(synsetfname, 'r').readlines()
    imageNetIDs = {}
    count = 0
    for categoryinfo in categorylist:
        wnetid = categoryinfo.split(' ')[0]
        categoryname = ' '.join(categoryinfo.split(' ')[1:])
        imageNetIDs[str(count)] = [wnetid, categoryname]
        count += 1

    assert len(imageNetIDs.keys()) == 1000
    return imageNetIDs

def getlabellist(synsetfname):
    """ read sysnset file as python list. Index corresponds to the output that 
    caffe provides
    """
    # 将 sysnset 文件读取为 python 列表。 索引对应caffe提供的输出 输出classlist
    
    categorylist = open(synsetfname, 'r').readlines()
    labellist = [category.split(' ')[0] for category in categorylist]
    return labellist


def compute_distance(query_channel, channel, mean_vec, distance_type = 'eucos'):
    """ Compute the specified distance type between chanels of mean vector and query image.
    In caffe library, FC8 layer consists of 10 channels. Here, we compute distance
    of distance of each channel (from query image) with respective channel of
    Mean Activation Vector. In the paper, we considered a hybrid distance eucos which
    combines euclidean and cosine distance for bouding open space. Alternatively,
    other distances such as euclidean or cosine can also be used. 
    
    Input:
    *****************************************************
    query_channel: Particular FC8 channel of query image
    channel: channel number under consideration
    mean_vec: mean activation vector

    Output:
   *******************************************************
    query_distance : Distance between respective channels

    """

    # 计算平均向量通道与查询图像之间的指定距离类型。
    #      在 caffe 库中，FC8 层由 10 个通道组成。 在这里，我们计算距离
    #      每个通道（来自查询图像）与相应通道的距离
    #      平均激活向量。 在本文中，我们考虑了一个混合距离 eucos，它
    #      结合欧几里得和余弦距离来包围开放空间。 或者，
    #      也可以使用其他距离，例如欧几里得或余弦。
    #
    #      输入：
    #      ****************************************************** ***
    #      query_channel：查询图像的特定 FC8 通道
    #      频道：考虑中的频道号
    #      mean_vec：平均激活向量
    #
    #      输出：
    #     ****************************************************** *****
    #      query_distance ：各个通道之间的距离
 
    query_channel = np.array(query_channel)
    mean_vec = np.reshape(mean_vec,(10,1))   # 这个10是不是需要改
    if distance_type == 'eucos':
        query_distance = spd.euclidean(mean_vec, query_channel)/200. + spd.cosine(mean_vec, query_channel)
    elif distance_type == 'euclidean':
        query_distance = spd.euclidean(mean_vec, query_channel)/200.
    elif distance_type == 'cosine':
        query_distance = spd.cosine(mean_vec, query_channel)
    else:
        print ("distance type not known: enter either of eucos, euclidean or cosine")
    return query_distance
    
