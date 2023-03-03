# This code defines several functions that use the 'pyplot' modiule from
# the 'matplotlib' library to plot and visualize data.

# Import the necessary libraries
from matplotlib import pyplot as plt

# Takes an array as input and creates a histogram of the array's values
def valueHistogram(array):
    flattenArray = [x for x in array.flatten("C")]
    plt.hist(x=flattenArray, bins='auto')

# Displays the array-input as a grayscale image
def standardPlot(array):
    plt.imshow(array, cmap='gray')
    plt.show()

# Takes two arrays as input and creates a plot with two subplots, sharing
# the same x-axis. 
def dualPlot(array1, array2):
    f, axarr = plt.subplots(2, sharex=True)
    axarr[0].plot(array1)
    axarr[1].plot(array2)
    plt.show()

# Takes three arrays as input and creates a plot with three subplots
# arranged in a row. Each subplot displays a grayscale image of one of
# the input arrays.
def tripplePlot(array1, array2, array3):
    plt.subplot(1,3,1)
    plt.imshow(array1, cmap='gray')
    plt.subplot(1,3,2)
    plt.imshow(array2, cmap='gray')
    plt.subplot(1,3,3)
    plt.imshow(array3, cmap='gray')
    plt.show()