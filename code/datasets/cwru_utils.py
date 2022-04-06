from os import listdir
import os
from scipy.io import loadmat
import numpy as np
import torch

def get_cwru_filepaths(root_dir,motor_load):

    filenames = {
        1 : '1.mat',
        2 : '2.mat',
        3 : '3.mat'
    }

    health_states = {
        'normal': 0,
        'b007':   1,
        'b014':   2,
        'b021':   3,
        'ir007':  4,
        'ir014':  5,
        'ir021':  6,
        'or007': 7,
        'or014': 8,
        'or021': 9,
    }

    class_dirs = listdir(root_dir)
    filepaths = []
    if '.DS_Store' in class_dirs:
        class_dirs.remove('.DS_Store')

    for class_dir in class_dirs:
        PATH = os.path.join(root_dir,class_dir,filenames[motor_load])
        filepaths.append((PATH,health_states[class_dir]))
        #print(PATH, class_dir, filenames[motor_load])
    return filepaths

def get_trainsamples(filepaths):
    """
    IN:
        list: filepaths pointing to training files
    OUT:
        list of tuples: (training_sample, label), sample and label are tensors.

    """
    trainsamplelist = []
    for filepath,label in filepaths:
        filedata = loadmat(filepath)
        print('\n',filepath)
        print("normal" in filepath, ' !')
        DE_arr = None
        FE_arr = None
        if 'normal' in filepath and '2.mat' in filepath:
            print("again: ",filepath)
            DE_arr = filedata['X099_DE_time']
            FE_arr = filedata['X099_FE_time']
        else:
            for key in filedata.keys():
                if 'DE' in key:
                    DE_arr = filedata[key]
                if 'FE' in key:
                    FE_arr = filedata[key]
        arr = np.hstack((DE_arr,FE_arr))
        print(arr.shape)

        signal_len = arr.shape[0]
        n_samples = 1980
        window_len = 2048
        stride = int((signal_len-window_len)/n_samples)
        print("stride: ",stride)
        start = 0
        stop = 2048
        for i in range(n_samples):

            #print(label)
            temp_arr = torch.from_numpy(np.copy(arr[start:stop,:].T)).float()
            trainsamplelist.append((temp_arr, torch.tensor(label)))
            start = start + stride
            stop = stop + stride
    print(trainsamplelist[0])
    print(len(trainsamplelist))

    return trainsamplelist


def get_testsamples(filepaths):
    """
    IN:
        list: filepaths pointing to training files
    OUT:
        list of tuples: (training_sample, label), sample and label are tensors.
    """
    testsamplelist = []
    for filepath,label in filepaths:
        filedata = loadmat(filepath)
        print('\n',filepath)
        print("normal" in filepath, ' !')
        DE_arr = None
        FE_arr = None
        if 'normal' in filepath and '2.mat' in filepath:
            print("again: ",filepath)
            DE_arr = filedata['X099_DE_time']
            FE_arr = filedata['X099_FE_time']
        else:
            for key in filedata.keys():
                if 'DE' in key:
                    DE_arr = filedata[key]
                if 'FE' in key:
                    FE_arr = filedata[key]
        arr = np.hstack((DE_arr,FE_arr))
        print(arr.shape)

        signal_len = arr.shape[0]
        n_samples = 59
        window_len = 2048
        stride = int((signal_len-window_len)/n_samples)
        if stride < 2048:
            stride = 2048
        print("stride: ",stride)
        start = 0
        stop = 2048
        for i in range(n_samples):
            #print(label)
            print(start, stop)
            #sample_window_as_tensor = torch.from_numpy(np.copy(loaded_data_from_matfile_numpy_array[start:stop,:].T)).float()
            temp_arr = torch.from_numpy(np.copy(arr[start:stop,:].T)).float()
            print(temp_arr.shape)
            if temp_arr.shape[1] != 2048:

                raise ValueError
            testsamplelist.append((temp_arr, torch.tensor(label)))
            start = start + stride
            stop = stop + stride

    print(testsamplelist[0])
    print(len(testsamplelist))

    return testsamplelist

def get_valfold_indices(n_samples, n_folds, n_classes):
    """
    IN:
    - (int) n_folds - number of validation folds in the dataset
    - (int) n_samples - number of samples in the dataset
    - (int) n_classes - number of classes in the dataset
    OUT:
    - (dict) fold_indice_dict - lists of boolean values pointing validation
      indices as True and training indices as False

     NOTE:
     - Each class is expected to have same amount of samples in the dataset.
     - Samples corresponding to same class are expected to be in a row in the
       dataset. I.e. Indexes of class 1 can't be between indexes of class 2.
    """
    og_starts = np.arange(0,n_samples,int(n_samples/n_classes))
    og_sample_indices = np.zeros((n_samples,1),dtype = bool)
    fold_indice_dict = {}
    for i in range(n_folds):
        fold_indices = np.copy(og_sample_indices)
        starts = og_starts+int(n_samples/n_classes/n_folds)*i
        for start in starts:
            for idx in range(start, start+int(n_samples/n_classes/n_folds),1):
                fold_indices[idx] = True
        fold_indice_dict[i] = fold_indices
    for i in range(n_folds):
        print('\n', i)
        print(fold_indice_dict[i])
        print('\n-----------------------------')
    return fold_indice_dict
