
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import os
from os import listdir
from sklearn.model_selection import train_test_split


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
            filepath = filepath[3:]
            if torch.cuda.is_available():
                filepath = '../original'+filepath[15:]
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


def get_random_thruster_filepaths(faults,vessel_dir, filter1, filter2):
    print(filter1, filter2)
    """Randomly divides data measured with Input shaft RAW sensor from thrusters 1,5 and 6
    in to train, validation and test datasets. These datasets hold
    50 % of faulty data and 50 % of healthy data.
    Each fault has its own train, validation and test dataset.

    IN:
    dict - fault dictionary {'fault type' : 'label index'}

    OUT:
    dict - {'fault_type':{'train':[*filepaths], 'val':[*filepaths], 'test': [*filepaths]}}"""

    thruster_filepaths = {}
    thruster_dict = get_thruster_files(vessel_dir)
    print(thruster_dict.keys())
    print('1: ',len(thruster_dict['1']['Input shaft RAW (rms)']))
    print('5: ',len(thruster_dict['5']['Input shaft RAW (rms)']))
    print('6: ',len(thruster_dict['6']['Input shaft RAW (rms)']))

    for fault, index in faults.items():

        thruster_filepaths[fault] = {}
        trainsplit, valsplit, testsplit = get_splits(index, thruster_dict, filter1, filter2)
        thruster_filepaths[fault]['train'] = trainsplit
        thruster_filepaths[fault]['val'] = valsplit
        thruster_filepaths[fault]['test'] = testsplit

    return thruster_filepaths

def get_thruster_files(vessel_dir):
    thruster_dict = {}
    thrusters = listdir(vessel_dir)
    rmlist = ['.DS_Store','T3','2','3','4']
    for item in rmlist:
        if item in thrusters:
            thrusters.remove(item)

    for thruster in thrusters:

        sensor_dict = {}
        sensors = listdir(os.path.join(vessel_dir,thruster))
        sensors = ['Input shaft RAW (rms)']
        if '.DS_Store' in sensors:
            sensors.remove('.DS_Store')
        for sensor in sensors:

            files = listdir(os.path.join(vessel_dir,thruster,sensor))
            if '.DS_Store' in files:
                files.remove('.DS_Store')


            files = [os.path.join(vessel_dir,thruster,sensor)+'/'+x for x in files]
            sensor_dict[sensor] = files
        thruster_dict[thruster] = sensor_dict
    return thruster_dict


def get_splits(faultkey, thruster_dict, filter1, filter2):

    faultyfilepaths = []
    healthyfilepaths = []
    all_files = []
    sensor = 'Input shaft RAW (rms)'
    t1hsum = 0
    t5hsum = 0
    t6hsum = 0
    t1fsum = 0
    t5fsum = 0
    t6fsum = 0
    filtered_files = 0

    t1hfiles = []
    t5hfiles = []
    t6hfiles = []
    t1ffiles = []
    t5ffiles = []
    t6ffiles = []
    for thruster in thruster_dict.keys():
        print(thruster)
        all_files+=thruster_dict[thruster][sensor]

    for file in all_files:
        label = file.strip('.csv').split('/')[-1].split('_')[-1]
        date = file.strip('.csv').split('/')[-1].split('_')[0].split(' ')[0]

        #print(date)
        thruster = file.strip('.csv').split('/')[-3]


        if label[faultkey] == '1':
            faultyfilepaths.append(file)
            if thruster == '1':
                t1fsum +=1
                t1ffiles.append(file)
            elif thruster == '5':
                if filter1 not in date and filter2 not in date:
                    t5fsum +=1
                    t5ffiles.append(file)
                else:
                    filtered_files+=1
            elif thruster == '6':
                if filter1 not in date and filter2 not in date:
                    t6fsum +=1
                    t6ffiles.append(file)
                else:
                    filtered_files+=1
        else:
            healthyfilepaths.append(file)
            if thruster == '1':
                t1hsum +=1
                t1hfiles.append(file)
            elif thruster == '5':
                if filter1 not in date and filter2 not in date:
                    t5hsum +=1
                    t5hfiles.append(file)
                else:
                    filtered_files+=1
            elif thruster == '6':
                if filter1 not in date and filter2 not in date:
                    t6hsum +=1
                    t6hfiles.append(file)
                else:
                    filtered_files+=1
    print('t1 healthy: ',t1hsum, len(t1hfiles))
    print('t5 healthy: ',t5hsum, len(t5hfiles))
    print('t6 healthy: ',t6hsum, len(t6hfiles))
    print('t1 faulty: ',t1fsum, len(t1ffiles))
    print('t5 faulty: ',t5fsum, len(t5ffiles))
    print('t6 faulty: ',t6fsum, len(t6ffiles))
    print("total healthy: ",len(healthyfilepaths))
    print("total faulty: ",len(faultyfilepaths))
    print("total samples: ",len(healthyfilepaths)+len(faultyfilepaths))
    print("total samples separated by thruster: ",len(t1hfiles)+len(t5hfiles)+len(t6hfiles)+len(t1ffiles)+len(t5ffiles)+len(t6ffiles))
    print("from total files: ",len(all_files))
    print("filtered_files: ",filtered_files)



    if len(healthyfilepaths) < len(faultyfilepaths):
        max_num_samples = len(healthyfilepaths)
    else:
        max_num_samples = len(faultyfilepaths)

    healthyfilepaths = np.random.choice(healthyfilepaths, size = max_num_samples, replace = False).tolist()
    faultyfilepaths = np.random.choice(faultyfilepaths, size = max_num_samples, replace = False).tolist()

    healthytrainset, healthytestset = train_test_split(healthyfilepaths, test_size = 0.33,shuffle = True)
    healthyvalset, healthytestset = train_test_split(healthytestset, test_size = 0.5,shuffle = True)

    faultytrainset, faultytestset = train_test_split(faultyfilepaths, test_size = 0.33,shuffle = True)
    faultyvalset, faultytestset = train_test_split(faultytestset, test_size = 0.5,shuffle = True)

    trainset = healthytrainset+faultytrainset
    print("balanced trainset: ",len(trainset))
    valset = healthyvalset+faultyvalset
    print("balanced valset: ",len(valset))
    testset = healthytestset+faultytestset
    print("balanced testset: ",len(testset))

    return trainset, valset, testset


if __name__ == '__main__':
    THRUSTER_root_dir = '../../original.tmp/labeled_data'
    faults = {'bearing': 4,
            'wheel': 9}

    filter1 = '2018-09'
    filter2 = '2018-10'

    thruster_filepaths = get_random_thruster_filepaths(faults,THRUSTER_root_dir,filter1,filter2)
    import json
    file = open('thruster_filepaths/fset0.json', 'w+')
    json.dump(thruster_filepaths, file)
    file.close()

    
