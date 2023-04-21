# This code defines two functions 'get2dHistograms' and 'dataToArray',
# which are used to load and manipulate data from an HDF5 file, using 
# the 'h5py' library and the 'numpy' array data structure.

# Import the necessary libraries
import h5py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display


def get2dHistograms(path):
    f = h5py.File(path)
    keys = list(f.keys())
    dataset = [f[key]["data"] for key in keys]
    return dataset

def dataToArray(path):
    return np.array(get2dHistograms(path))


def plot_metrics(augmentation_types, all_metrics):
    num_types = len(augmentation_types)
    
    metric_names = list(all_metrics[0].keys())
    num_metrics = len(metric_names)

    plt.figure(figsize=(20, 20))
    
    plot_index = 1
    for metric_name in metric_names:
        plt.subplot(num_metrics//2, 2, plot_index)
        for i, augment_type in enumerate(augmentation_types):
            plt.plot(all_metrics[i][metric_name], label=f'{augment_type} {metric_name}')
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel(metric_name)
        plt.title(f'{metric_name} with Different Augmentations')
        plot_index += 1

    plt.tight_layout()
    plt.show()

def display_metrics_table(augmentation_types, all_metrics):
    data = []
    columns = ["Augmentation", "Run", "Epoch", "Train Loss", "Train Acc", "Test Loss", "Test Acc"]

    for aug_index, augment_type in enumerate(augmentation_types):
        for run in range(len(all_metrics[aug_index])):
            for epoch in range(len(all_metrics[aug_index]['train_losses'])):
                data.append([augment_type,
                             run + 1,
                             epoch + 1,
                             all_metrics[aug_index]['train_losses'][epoch],
                             all_metrics[aug_index]['train_accs'][epoch],
                             all_metrics[aug_index]['test_losses'][epoch],
                             all_metrics[aug_index]['test_accs'][epoch]])

    df = pd.DataFrame(data, columns=columns)
    display(df)