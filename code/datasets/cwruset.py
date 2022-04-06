from torch.utils.data import Dataset
#from cwru_utils import time_window_division, get_cwru_filepaths, get_trainsets, get_testsets
import numpy as np
#from cwru_utils import get_cwru_filepaths, get_trainsamples, get_testsamples, get_valfold_indices


class ValidationSet(Dataset):
    def __init__(self,windows_and_labels):
        self.validation_set = windows_and_labels

    def __len__(self):
        return len(self.validation_set)

    def __getitem__(self,idx):
        return self.validation_set[idx]


class CWRUDataset(Dataset):
    """
    Class for bearing fault dataset with 10 health conditions.
    This class loads data from all conditions corresponding to a CWRU load
    domain, I.e. 1, 2 and 3 hp.

    The class divides the loaded data to time windows of 2048 time steps.

    This class can function as a training set or a test set.
    If the class is a training set, then a validation fold can be reserved and
    changed during training. Furthermore, the training time windows overlap.

    If the class is a test set, the time windows divided from the loaded data
    do not overlap.

    For test and train use there are an equal amount of time windows from each
    loaded data.

    """
    def __init__(self, args, root_dir, motor_load = 1, train_set = True, use_val_folds = False):
        self.root_dir = root_dir
        self.motor_load = motor_load
        self.train_set = train_set
        self.filepaths = get_cwru_filepaths(self.root_dir, self.motor_load)

        self.use_val_folds = use_val_folds

        if train_set:
            self.trainsamplelist = get_trainsamples(self.filepaths, args)
            self.testsamplelist = None
            if self.use_val_folds:
                # TODO make it work on other n_samples
                self.folds = 5
                self.validation_fold = 0
                self.fold_indice_dict = get_valfold_indices(len(self.trainsamplelist),self.folds,10)
                self.iterable_trainset = None
                self.val_sample_list = None
                self.change_validation_fold()
        else:
            self.trainsamplelist = None
            self.testsamplelist = get_testsamples(self.filepaths, args)

    def __len__(self):
        if self.train_set:
            if self.use_val_folds:
                return len(self.iterable_trainset)
            else:
                return len(self.trainsamplelist)
        else:
            return len(self.testsamplelist)

    def __getitem__(self,idx):
        if self.train_set:
            if self.use_val_folds:
                return self.iterable_trainset[idx]
            else:
                return self.trainsamplelist[idx]
        else:
            return self.testsamplelist[idx]
    #
    def get_validation_dataset(self):
        validation_dataset = ValidationSet(self.val_sample_list)
        return validation_dataset
    #
    def change_validation_fold(self):

        fold_indices = self.fold_indice_dict[self.validation_fold]
        self.iterable_trainset = [x for x, is_val_sample in zip(self.trainsamplelist,fold_indices) if not is_val_sample]
        self.val_sample_list = [x for x, is_val_sample in zip(self.trainsamplelist,fold_indices) if is_val_sample]

    #
    def next_fold(self):
        if self.validation_fold == self.folds-1:
            self.validation_fold = 0
        else:
            self.validation_fold +=1
        print("validation fold now: ", self.validation_fold)
        self.change_validation_fold()
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

def get_trainsamples(filepaths, args):
    """
    IN:
        list: filepaths pointing to training files
    OUT:
        list of tuples: (training_sample, label), sample and label are tensors.

    """
    trainsamplelist = []
    for filepath,label in filepaths:
        filedata = loadmat(filepath)
        #print('\n',filepath)
        #print("normal" in filepath, ' !')
        DE_arr = None
        FE_arr = None
        if 'normal' in filepath and '2.mat' in filepath:
            #print("again: ",filepath)
            DE_arr = filedata['X099_DE_time']
            FE_arr = filedata['X099_FE_time']
        else:
            for key in filedata.keys():
                if 'DE' in key:
                    DE_arr = filedata[key]
                if 'FE' in key:
                    FE_arr = filedata[key]
        if args.DE_FE == True:
            arr = np.hstack((DE_arr,FE_arr))
        else:
            arr = DE_arr
        #arr = FE_arr
        #print(arr.shape)

        signal_len = arr.shape[0]
        n_samples = 1980
        window_len = 2048
        stride = int((signal_len-window_len)/n_samples)
        #print("stride: ",stride)
        start = 0
        stop = 2048
        for i in range(n_samples):

            #print(label)
            temp_arr = torch.from_numpy(np.copy(arr[start:stop,:].T)).float()
            trainsamplelist.append((temp_arr, torch.tensor(label)))
            start = start + stride
            stop = stop + stride
    #print(trainsamplelist[0])
    #print(len(trainsamplelist))

    return trainsamplelist


def get_testsamples(filepaths, args):
    """
    IN:
        list: filepaths pointing to training files
    OUT:
        list of tuples: (training_sample, label), sample and label are tensors.
    """
    testsamplelist = []
    for filepath,label in filepaths:
        filedata = loadmat(filepath)
        #print('\n',filepath)
        #print("normal" in filepath, ' !')
        DE_arr = None
        FE_arr = None
        if 'normal' in filepath and '2.mat' in filepath:
            #print("again: ",filepath)
            DE_arr = filedata['X099_DE_time']
            FE_arr = filedata['X099_FE_time']
        else:
            for key in filedata.keys():
                if 'DE' in key:
                    DE_arr = filedata[key]
                if 'FE' in key:
                    FE_arr = filedata[key]
        if args.DE_FE == True:
            arr = np.hstack((DE_arr,FE_arr))
        else:
            arr = DE_arr
        #arr = FE_arr
        #print(arr.shape)

        signal_len = arr.shape[0]
        n_samples = 59
        window_len = 2048
        stride = int((signal_len-window_len)/n_samples)
        if stride < 2048:
            stride = 2048
        #print("stride: ",stride)
        start = 0
        stop = 2048
        for i in range(n_samples):
            #print(label)
            #print(start, stop)
            #sample_window_as_tensor = torch.from_numpy(np.copy(loaded_data_from_matfile_numpy_array[start:stop,:].T)).float()
            temp_arr = torch.from_numpy(np.copy(arr[start:stop,:].T)).float()
            #print(temp_arr.shape)
            if temp_arr.shape[1] != 2048:

                raise ValueError
            testsamplelist.append((temp_arr, torch.tensor(label)))
            start = start + stride
            stop = stop + stride

    #print(testsamplelist[0])
    #print(len(testsamplelist))

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
    # for i in range(n_folds):
        # print('\n', i)
        # print(fold_indice_dict[i])
        # print('\n-----------------------------')
    return fold_indice_dict

if __name__ == '__main__':


    dataset = CWRUDataset(root_dir = '../../original.tmp/12k/drive_end2/',motor_load = 3, train_set = True, use_val_folds = True)
    print(dataset.__len__())
    for i in range(5):
        valset = dataset.get_validation_dataset()
        dataset.next_fold()
    valset = dataset.get_validation_dataset()
    dataset.next_fold()
    # for i in range(dataset.__len__()):
    #     print(dataset.__getitem__(i))
