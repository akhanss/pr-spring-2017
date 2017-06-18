#!/usr/bin/env python

"""
Builds image data base as test, train, validatation datasets
Run script as python create_images.py $mode
where mode can be 'test', 'train', 'val'

"""

import sys

from joblib import Parallel, delayed

import pickle

import numpy as np
import pandas as pd

import os
import glob

from PIL import Image

from sklearn.cross_validation import train_test_split

import SimpleITK as sitk

raw_image_path = '../../data/raw/*/'
candidates_file = '../data/candidates.csv'


class CTScan(object):
    """
	A class that allows you to read .mhd header data, crop images and 
	generate and save cropped images

    Args:
    filename: .mhd filename
    coords: a numpy array
	"""
    def __init__(self, filename = None, coords = None, path = None):
        """
        Args
        -----
        filename: .mhd filename
        coords: coordinates to crop around
        ds: data structure that contains CT header data like resolution etc
        path: path to directory with all the raw data
        """
        self.filename = filename
        self.coords = coords
        self.ds = None
        self.image = None
        self.path = path

    def reset_coords(self, coords):
        """
        updates to new coordinates
        """
        self.coords = coords

    def read_mhd_image(self):
        """
        Reads mhd data
        """
        path = glob.glob(self.path + self.filename + '.mhd')
        self.ds = sitk.ReadImage(path[0])
        self.image = sitk.GetArrayFromImage(self.ds)

    def get_voxel_coords(self):
        """
        Converts cartesian to voxel coordinates
        """
        origin = self.ds.GetOrigin()
        resolution = self.ds.GetSpacing()
        voxel_coords = [np.absolute(self.coords[j]-origin[j])/resolution[j] \
            for j in range(len(self.coords))]
        return tuple(voxel_coords)
    
    def get_image(self):
        """
        Returns axial CT slice
        """
        return self.image
    
    def get_subimage(self, width):
        """
        Returns cropped image of requested dimensiona
        """
        self.read_mhd_image()
        x, y, z = self.get_voxel_coords()
        subImage = self.image[int(z), int(y-width/2):int(y+width/2),\
         int(x-width/2):int(x+width/2)]
        return subImage   
    
    def normalizePlanes(self, npzarray):
        """
        Copied from SITK tutorial converting Houndsunits to grayscale units
        """
        maxHU = 400.
        minHU = -1000.
        npzarray = (npzarray - minHU) / (maxHU - minHU)
        npzarray[npzarray>1] = 1.
        npzarray[npzarray<0] = 0.
        return npzarray
    
    def save_image(self, filename, width):
        """
        Saves cropped CT image
        """
        image = self.get_subimage(width)
        image = self.normalizePlanes(image)
        Image.fromarray(image*255).convert('L').save(filename)


def create_data(idx, outDir, X_data,  width = 50):
    '''
    Generates your test, train, validation images
    outDir = a string representing destination
    width (int) specify image size
    '''
    try:
      scan = CTScan(np.asarray(X_data.loc[idx])[0], \
          np.asarray(X_data.loc[idx])[1:], raw_image_path)
      outfile = outDir  +  str(idx)+ '.jpg'
      scan.save_image(outfile, width)
    except:
      print(str(idx))

def do_test_train_split(filename):
    """
    Does a test train split if not previously done

    """
    candidates = pd.read_csv("../data/candidates.csv")

    positives = candidates[candidates['class']==1].index  
    negatives = candidates[candidates['class']==0].index

    ## Under Sample Negative Indexes
    
    np.random.seed(42)
    negIndexes = np.random.choice(negatives, len(positives)*5, replace = False)
    
    candidatesDf = candidates.iloc[list(positives)+list(negIndexes)]
    
    X = candidates.iloc[:,:-1]
    y = candidates.iloc[:,-1]
    # Augumentation
    for i in range(0, 2):
      positives = positives.append(positives)

    X1 = []; X2 =[]; X3 = []; X4 = []; X5 = []; XTest = []
    y1 = []; y2 = []; y3 = []; y4 = []; y5 = []; yTest = []
    for i in range(0, 900):
      X1.append(candidates.iloc[positives[i], :-1])
      y1.append(candidates.iloc[positives[i], -1])
    for i in range(900, 1800):
      X2.append(candidates.iloc[positives[i], :-1])
      y2.append(candidates.iloc[positives[i], -1])
    for i in range(1800, 2700):
      X3.append(candidates.iloc[positives[i], :-1])
      y3.append(candidates.iloc[positives[i], -1])
    for i in range(2700, 3600):
      X4.append(candidates.iloc[positives[i], :-1])
      y4.append(candidates.iloc[positives[i], -1])
    for i in range(3600, 4500):
      X5.append(candidates.iloc[positives[i], :-1])
      y5.append(candidates.iloc[positives[i], -1])
    for i in range(4500, 5404):
      XTest.append(candidates.iloc[positives[i], :-1])
      yTest.append(candidates.iloc[positives[i], -1])

    for i in range(0, 916):
      X1.append(candidates.iloc[negatives[i], :-1])
      y1.append(candidates.iloc[negatives[i], -1])
    for i in range(916, 1832):
      X2.append(candidates.iloc[negatives[i], :-1])
      y2.append(candidates.iloc[negatives[i], -1])
    for i in range(1832, 2748):
      X3.append(candidates.iloc[negatives[i], :-1])
      y3.append(candidates.iloc[negatives[i], -1])
    for i in range(2748, 3664):
      X4.append(candidates.iloc[negatives[i], :-1])
      y4.append(candidates.iloc[negatives[i], -1])
    for i in range(3664, 4580):
      X5.append(candidates.iloc[negatives[i], :-1])
      y5.append(candidates.iloc[negatives[i], -1])
    for i in range(4580, 5497):
      XTest.append(candidates.iloc[negatives[i], :-1])
      yTest.append(candidates.iloc[negatives[i], -1])

    X1 = pd.DataFrame(X1)
    X2 = pd.DataFrame(X2)
    X3 = pd.DataFrame(X3)
    X4 = pd.DataFrame(X4)
    X5 = pd.DataFrame(X5)
    XTest = pd.DataFrame(XTest)

    y1 = pd.DataFrame(y1)
    y2 = pd.DataFrame(y2)
    y3 = pd.DataFrame(y3)
    y4 = pd.DataFrame(y4)
    y5 = pd.DataFrame(y5)
    yTest = pd.DataFrame(yTest)   

    """
    X_train, X_test, y_train, y_test = train_test_split(X, y,\
     test_size = 0.20, random_state = 42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, \
        test_size = 0.20, random_state = 42)
    """
    X1.to_pickle('train1')
    y1.to_pickle('trainlabels1')

    X2.to_pickle('train2')
    y2.to_pickle('trainlabels2')

    X3.to_pickle('train3')
    y3.to_pickle('trainlabels3')

    X4.to_pickle('train4')
    y4.to_pickle('trainlabels4')

    X5.to_pickle('train5')
    y5.to_pickle('trainlabels5')

    XTest.to_pickle('test')
    yTest.to_pickle('testlabels')
    """
    X_train.to_pickle('traindata')
    y_train.to_pickle('trainlabels')
    X_test.to_pickle('testdata')
    y_test.to_pickle('testlabels')
    X_val.to_pickle('valdata')
    y_val.to_pickle('vallabels')
    """

def main():
    """
    if len(sys.argv) < 2:
        raise ValueError('1 argument needed. Specify if you need to generate a train, test or val set')
    else:
        mode = sys.argv[1]
        if mode not in ['train', 'test', 'val']:
            raise ValueError('Argument not recognized. Has to be train, test or val')
    """
    mode = 'cross'
    inpfile = mode + 'data'
    outDir = mode + '/image_'
    do_test_train_split(candidates_file)
    """
    if os.path.isfile(inpfile):
        pass
    else:
        do_test_train_split(candidates_file)
    
    X_data = pd.read_pickle(inpfile)
    Parallel(n_jobs = 3)(delayed(create_data)(idx, outDir, X_data) for idx in X_data.index)
    """

    X1_data = pd.read_pickle("train1")
    X2_data = pd.read_pickle("train2")
    X3_data = pd.read_pickle("train3")
    X4_data = pd.read_pickle("train4")
    X5_data = pd.read_pickle("train5")
    XT_data = pd.read_pickle("test")

    Parallel(n_jobs = 10)(delayed(create_data)(idx, "../data/cross/1/image_", X1_data) for idx in X1_data.index)
    Parallel(n_jobs = 10)(delayed(create_data)(idx, "../data/cross/2/image_", X2_data) for idx in X2_data.index)
    Parallel(n_jobs = 10)(delayed(create_data)(idx, "../data/cross/3/image_", X3_data) for idx in X3_data.index)
    Parallel(n_jobs = 10)(delayed(create_data)(idx, "../data/cross/4/image_", X4_data) for idx in X4_data.index)
    Parallel(n_jobs = 10)(delayed(create_data)(idx, "../data/cross/5/image_", X5_data) for idx in X5_data.index)
    Parallel(n_jobs = 10)(delayed(create_data)(idx, "../data/cross/test/image_", XT_data) for idx in XT_data.index)
    

if __name__ == "__main__":
    main()

        