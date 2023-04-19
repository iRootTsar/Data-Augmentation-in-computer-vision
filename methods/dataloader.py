# This code defines two functions 'get2dHistograms' and 'dataToArray',
# which are used to load and manipulate data from an HDF5 file, using 
# the 'h5py' library and the 'numpy' array data structure.

# Import the necessary libraries
import h5py
import numpy as np


def get2dHistograms(path):
    f = h5py.File(path)
    keys = list(f.keys())
    dataset = [f[key]["data"] for key in keys]
    return dataset

def dataToArray(path):
    return np.array(get2dHistograms(path))