import matplotlib.pyplot as plt
from collections import OrderedDict
import datetime
import numpy as np
from scipy.io import loadmat
import torch
import os
from os import listdir

def get_cwru_filepaths(root_dir,health_states,motor_load):
    """
    Utility function that fetches all data filepaths corresponding
    to the motor load used during testbench measurements.

    root_dir needs to be of the format:
    '../original.tmp/12k/'

    Returns a list of tuples (filepath, healt state).
    """

    #filename dict corresponding to the motor load parametere
    filenames = {
        1 : ['1.mat'],
        2 : ['2.mat'],
        3 : ['3.mat'],
        4 : ['1.mat','2.mat','3.mat']
    }
    #class directories correspond to labels
    class_dirs = listdir(root_dir)
    filepaths = []
    if '.DS_Store' in class_dirs:
        class_dirs.remove('.DS_Store')
    for class_dir in class_dirs:
        PATH = os.path.join(root_dir,class_dir)
        if 'unnecessary' not in PATH:

            #Choose the correct file corresponding to the given motor load
            temp_path = PATH # if motor_load = 4
            for filename in filenames[motor_load]:
                PATH = temp_path + '/' + filename

                filepaths.append((PATH,health_states[class_dir]))

    return filepaths

def get_trainsets(filepaths, normalize, balance, len_fold):

    """
    A utility function that reads each file with

    """
    # From every 10 files, select overlapping test windows of lenght 2048.
    #Choose overlapping windows from the  first 50% of the sequence.
    # Test data windows are selected from the last 50% of the sequence.
    # Return len_fold amount of window sets.

    #print("getting train folds for:",filepaths[0][1])

    window_len = 2048
    stride = 32 #overlap

    fold_dict = {}
    for i in range(len_fold):
        fold_dict[i] = []

    num_windows = 1834# prior knowledge, that this is max number for balanced dataset if stride = 32 and wlen = 2048

    # go through each file with a vibration sequence per health state
    for filepath in filepaths:

        path, label = filepath[0], filepath[1]
        data = loadmat(path)
        array = get_array(data)

        train_length = 60705 # roughly the first 50 % of the timeseries signal

        array = np.copy(array[:train_length,:])

        if normalize:

            array = normalize_arr(array)


        for i in range(len_fold):

            start_p, stop_p = i/len_fold, (i+1)/len_fold
            start, stop = int(train_length*start_p), int(train_length*stop_p)

            window_start, window_end = start, start + 2048

            #No overlap between folds!
            while not window_end > stop :

                fold_dict[i].append((array[window_start:window_end,:],label))
                window_start, window_end = window_start+32, window_end +32


    train_folds = []

    for i in range(len_fold):
        train_folds.append(fold_dict[i])
    return train_folds

def get_testset(filepaths,normalize,balance):
    #from every 10 files, select non-overlapping test windows of lenght 2048'
    #Choose non-overlapping windows from the last 50 % of the sequnence
    # Training (also validation) data windows are selected from the first 50 %
    # of the sequence

    num_windows = 29 #prior knowledge

    testset = []

    for filepath in filepaths:
        path, label = filepath[0], filepath[1]
        data = loadmat(path)
        array = get_array(data)
        test_length = int(0.5*array.shape[0])
        array = np.copy(array[-test_length:,:])
        if normalize:
            array = normalize_arr(array)


        for i in range(num_windows):
            start, stop =2048*i,2048*(i+1)
            testset.append((torch.from_numpy(array[start:stop,:].T).float(),label))

    return testset

def get_array(data):
    drive_end_key = 'DE'
    fan_end_key  = 'FE'
    de_arr = None
    fe_arr = None
    for key in data.keys():
        if drive_end_key in key:
            de_arr = data[key]
        if fan_end_key in key:
            fe_arr = data[key]
    #print(de_arr.shape,fe_arr.shape)
    #print(de_arr.shape,fe_arr.shape)
    arr = np.hstack((de_arr,fe_arr))

    return arr

def normalizearr1d(arr1d):
    """Normalizes array to the range [-1,1]"""

    arr1d_min = np.min(arr1d)

    arr1d_max = np.max(arr1d)

    arr1d = 2 * (arr1d - arr1d_min) / (arr1d_max - arr1d_min) - 1
    return arr1d

def normalize_arr(window):

    for i in range(window.shape[1]):

        window[:,i] = normalizearr1d(window[:,i].copy())

    return window

def save_cwru_history(history,SAVEPATH,model_name,motor_load):
    fig,axes = plt.subplots()
    l1, = axes.plot(history['train'],label = "Training error")
    l2, = axes.plot(history['val'], label = "Validation error")
    plt.legend(handles=[l1,l2])
    axes.set_title('Convergence of '+model_name)
    axes.set_ylabel('CrossEntropyLoss')
    axes.set_xlabel('Epoch')
    fig.savefig(SAVEPATH+model_name+'-'+motor_load+'.png')
    plt.close(fig)

def save_path_formatter(args):
    args_dict = vars(args)
    data_folder_name = args_dict['dataset']
    folder_string = []

    key_map = OrderedDict()
    key_map['arch'] =''
    key_map['batch_size']='bs'
    key_map['deconv']='deconv'
    key_map['delinear']='delinear'
    key_map['batchnorm'] = 'bn'

    key_map['lr']='lr'
    key_map['stride']='stride'
    key_map['eps'] = 'eps'
    key_map['deconv_iter'] = 'it'
    #key_map['lr_scheduler']=''

    key_map['epochs'] = 'ep'




    key_map['bias'] = 'bias'
    #key_map['block_fc']='bfc'
    #key_map['freeze']='freeze'


    for key, key2 in key_map.items():
        value = args_dict[key]
        if key2 is not '':
            folder_string.append('{}.{}'.format(key2, value))
        else:
            folder_string.append('{}'.format(value))

    save_path = ','.join(folder_string)
    timestamp = datetime.datetime.now().strftime("%m-%d-%H.%M")
    return os.path.join('./logs',data_folder_name,save_path,timestamp).replace("\\","/")
