import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from cwru_utils import get_cwru_filepaths, get_trainsets, get_testset

###------- Thruster -------

class Fast_Thrusterset(Dataset):
    def __init__(self,filepaths, args, stride = False):
        self.filepaths = filepaths
        if stride:
            self.stride = 32
        else:
            self.stride = 2048

        self.nstartsperfile = int(2048/self.stride)+1
        self.starts = [x*self.stride for x in range(self.nstartsperfile)]

        self.datalist = []
        print("initialising dataset")
        self.init_datalist()

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self,idx):
        return self.datalist[idx]

    def init_datalist(self):
        
        for filepath in self.filepaths:
            date, condition = extract_date_n_condition(filepath)
            for start in self.starts:
                stop = start + 2048
                array = self.extract_seq(filepath,start, stop)
                dictionary_entry = {'data':array, 'labels':condition}
                self.datalist.append(dictionary_entry)

        print("dataset_initialised")
        print("data len:", self.__len__())


    def extract_seq(self,filepath, start, stop):
        df = pd.read_csv(filepath, header = 0, index_col = 0)
        data = df.to_numpy()
        data = data[start:stop]
        data = torch.from_numpy(data.T).float()
        return data



class Thrusterset(Dataset):
    """
    An iterable dataset with three interface functions:
    __init__(*args) initialises the dataset according to the given arguments
    IN:
    filepaths - list of paths to timeseries data with labels in their filenames
    args - arguments given to the program


    __len__() returns the number of train / test samples in the dataset
    OUT:
    integer - number of samples in the dataset


    __getitem__(idx) returns a train / test sample from the dataset:
    IN:
    integer - index of a sample

    OUT:
    dict - {'data': (2048,1), 'labels': integer, 'date': timestamp of sample }
    """
    def __init__(self,filepaths, args, stride = False):
        self.filepaths = filepaths
        if stride:
            self.stride = 32
        else:
            self.stride = 2048
        self.nstartsperfile = int(2048/self.stride)+1
        self.repeatedfilepaths = self.filepaths*self.nstartsperfile
        self.starts = [x*self.stride for x in range(self.nstartsperfile)]

    def __len__(self):
        return(len(self.repeatedfilepaths))

    def __getitem__(self,idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        start, stop = self.get_start(idx)
        filepath = self.repeatedfilepaths[idx]
        data = self.extract_seq(filepath,start,stop)
        date,condition = extract_date_n_condition(filepath)
        sample = {'data':data,'labels':condition, 'date':date}
        return sample

    def get_start(self,idx):
        start_idx = int(idx/len(self.filepaths))
        return self.starts[start_idx], self.starts[start_idx]+2048

    def extract_seq(self,filepath, start, stop):
        df = pd.read_csv(filepath, header = 0, index_col = 0)
        data = df.to_numpy()
        data = data[start:stop]
        data = torch.from_numpy(data.T).float()
        return data

def extract_date_n_condition(filepath):

    date_n_condition = filepath.split('/')[-1].strip('.csv')
    date, condition = date_n_condition.split('_')
    condition = make_array(condition)

    return date, condition

def make_array(condition):
    list1=[]

    list1[:0]=condition
    interesting_indices = [4,8,9]
    list2 = [list1[x] for x in interesting_indices]
    list2 = np.array(list2, dtype = np.float32)
    list2 = torch.tensor(list2)
    return list2


###------- CWRU --------


health_states = {
    'normal': 0,
    'b007':   1,
    'b014':   2,
    'b021':   3,
    'ir007':  4,
    'ir014':  5,
    'ir021':  6,
    'or007_6': 7,
    'or014_6': 8,
    'or021_6': 9,
}
motor_loads = {
    'A' : 1,
    'B' : 2,
    'C' : 3,
    'D' : 4
}

class ValidationSet(Dataset):
    def __init__(self,windows_and_labels):
        self.validation_set = self.make_set(windows_and_labels)

    def __len__(self):
        return len(self.validation_set)

    def __getitem__(self,idx):
        return self.validation_set[idx]


    def make_set(self,windows_and_labels):
        temp_list = []
        for window,label in windows_and_labels:
            window = torch.from_numpy(window.T).float()
            temp_list.append((window,label))
        return temp_list

class CWRUDataset(Dataset):
    """Class for bearing fault dataset. The data holds 10 health conditions:

    0: healthy
    1-3: Ball faults (with severitys 0.07, 0.14, 0.21 inch)
    4-6: Inner ring faults (with severitys 0.07, 0.14, 0.21 inch)
    7-9: Outer ring faults (with severitys 0.07, 0.14, 0.21 inch),

    There are three different loads for each 10 health conditions:
    A: 1 hp
    B: 2 hp
    C: 3 hp
    D: 1,2,3 hp

    Each instant of this dataset class holds an ite"""

    def __init__(self,root_dir,motor_load='A',train_set = True, normalize = False, balance = True):
        self.root_dir = root_dir

        self.motor_load = motor_loads[motor_load]

        self.train_set = train_set
        self.filepaths = get_cwru_filepaths(self.root_dir,health_states,self.motor_load)

        self.folds = 5 # Change for n-fold cross validation

        if train_set:
            # 316 windows per fold and label with 5 fold CV
            self.trainsets = get_trainsets(self.filepaths, normalize, balance, self.folds)# list of lists of tuples (window, label)
            self.testset =None
        else:
            self.trainsets = None
            self.testset = get_testset(self.filepaths,normalize,balance) #list of tuples (window,label)
        self.validation_fold = 0 #range of values from 0 to len(self.folds)
        self.iterable_trainset = None

        if train_set:
            self.change_validation_fold()

    def __get_validation_dataset__(self):
        validation_dataset = ValidationSet(self.trainsets[self.validation_fold])
        return validation_dataset

    def __len__(self):
        if self.train_set:
            return len(self.iterable_trainset)
        else:
            return len(self.testset)

    def __getitem__(self,idx):
        """Returns tuple: (time window, health_state)"""
        #Todo torch tensors

        if self.train_set:
            return self.iterable_trainset[idx]
        else:
            return self.testset[idx]

    def change_validation_fold(self):
        """Puts all window-label tuples from train folds to the iterable_trainset
        used for training. Window-label tuples from validation_fold are excluded from
        changed iterable_trainset"""
        trainlist = []
        for fold in range(self.folds):
            if fold != self.validation_fold:
                #(len(self.trainsets))

                for sample in self.trainsets[fold]:
                    window, label = sample
                    window = torch.from_numpy(window.T).float()

                    trainlist.append((window,label))
        #print(len(trainlist))
        self.iterable_trainset = trainlist

    def next_fold(self):
        if self.validation_fold == 4:
            self.validation_fold = 0
        else:
            self.validation_fold +=1
        self.change_validation_fold()

    def print_fold(self):
        print(self.validation_fold)
