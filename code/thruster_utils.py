import os
from os import listdir
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

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


def get_splits(faultkey, thruster_dict, args):

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
                if args.filter1 not in date and args.filter2 not in date:
                    t5fsum +=1
                    t5ffiles.append(file)
                else:
                    filtered_files+=1
            elif thruster == '6':
                if args.filter1 not in date and args.filter2 not in date:
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
                if args.filter1 not in date and args.filter2 not in date:
                    t5hsum +=1
                    t5hfiles.append(file)
                else:
                    filtered_files+=1
            elif thruster == '6':
                if args.filter1 not in date and args.filter2 not in date:
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


def get_random_thruster_filepaths(faults,vessel_dir, args):
    print(args.filter1, args.filter2)
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
        trainsplit, valsplit, testsplit = get_splits(index, thruster_dict, args)
        thruster_filepaths[fault]['train'] = trainsplit
        thruster_filepaths[fault]['val'] = valsplit
        thruster_filepaths[fault]['test'] = testsplit

    return thruster_filepaths

def get_splits_by_thruster(faultkey, thruster_dict, train, test):

    faulty_trainfilepaths = []
    healthy_trainfilepaths = []
    all_trainfiles = []
    faulty_testfilepaths = []
    healthy_testfilepaths = []
    all_testfiles = []

    sensor = 'Input shaft RAW (rms)'
    for thruster in thruster_dict.keys():
        if thruster in train:
            all_trainfiles+=thruster_dict[thruster][sensor]
        elif thruster in test:
            all_testfiles += thruster_dict[thruster][sensor]

    for file in all_trainfiles:
        label = file.strip('.csv').split('/')[-1].split('_')[-1]
        if label[faultkey] == '1':
            faulty_trainfilepaths.append(file)
        else:
            healthy_trainfilepaths.append(file)

    for file in all_testfiles:
        label = file.strip('.csv').split('/')[-1].split('_')[-1]
        if label[faultkey] == '1':
            faulty_testfilepaths.append(file)
        else:
            healthy_testfilepaths.append(file)

    if len(healthy_trainfilepaths) < len(faulty_trainfilepaths):
        max_num_trainsamples = len(healthy_trainfilepaths)
    else:
        max_num_trainsamples = len(faulty_trainfilepaths)

    #print("len(healthy_testfilepaths)",len(healthy_testfilepaths))
    #print("len(faulty_testfilepaths)",len(faulty_testfilepaths))

    if len(healthy_testfilepaths) < len(faulty_testfilepaths):
        max_num_testsamples = len(healthy_testfilepaths)
    else:
        max_num_testsamples = len(faulty_testfilepaths)


    #clip to get balanced dataset
    healthy_trainfilepaths = np.random.choice(healthy_trainfilepaths, size = max_num_trainsamples, replace = False).tolist()
    faulty_trainfilepaths = np.random.choice(faulty_trainfilepaths, size = max_num_trainsamples, replace = False).tolist()

    healthy_testfilepaths = np.random.choice(healthy_testfilepaths, size = max_num_testsamples, replace = False).tolist()
    faulty_testfilepaths = np.random.choice(faulty_testfilepaths, size = max_num_testsamples, replace = False).tolist()

    #Divide trainsets to train and validation sets
    healthytrainset, healthyvalset = train_test_split(healthy_trainfilepaths, test_size = 0.33,shuffle = True)
    faultytrainset, faultyvalset = train_test_split(faulty_trainfilepaths, test_size = 0.33,shuffle = True)

    #join the faulty and healthy datasets
    trainset = healthytrainset+faultytrainset
    valset = healthyvalset+faultyvalset
    testset = healthy_testfilepaths+faulty_testfilepaths

    return trainset, valset, testset

def get_separated_thruster_filepaths(faults,vessel_dir, train=['1','5'],test = ['6']):
    """Randomly divides data measured with Input shaft RAW sensor from thrusters
    given as input in to train and validation dataset. Testset consists of data
    measured with Input shaft RAW sensor measured from thrusters listed in the 'test'
    input parameter. Train, validation and test datasets hold 50 % of faulty and
    50 % of healthy samples.

    IN:
    train - list[*thruster names]
    test - list[*thruster names]
    dict - fault dictionary {'fault type' : 'label index'}

    OUT:
    dict - {'fault_type':{'train':[*filepaths], 'val':[*filepaths], 'test': [*filepaths]}}
    """

    thruster_filepaths = {}
    thruster_dict = get_thruster_files(vessel_dir)
    #print(thruster_dict['1'].keys())

    for fault, index in faults.items():
        #print(fault, index)
        thruster_filepaths[fault] = {}
        trainsplit,valsplit, testsplit = get_splits_by_thruster(index, thruster_dict, train, test)
        thruster_filepaths[fault]['train'] = trainsplit
        thruster_filepaths[fault]['val'] = valsplit
        thruster_filepaths[fault]['test'] = testsplit

    return thruster_filepaths

def get_splits_sensor_thruster(faultkey, thruster_dict):
    faultyfilepaths = []
    healthyfilepaths = []
    all_files = []
    faulty_testfilepaths = []
    healthy_testfilepaths = []
    all_testfilepaths = []
    test_sensors = {4:'Pinion shaft gear end RAW',
                    8:'Pinion shaft coupling end RAW',
                    9:'propeller-shaft-fwd-end-raw'}
    test_thrusters = {4:'1',
                        8:'5',
                        9:'6'}
    test_sensor = test_sensors[faultkey]
    test_thruster = test_thrusters[faultkey]

    sensor = 'Input shaft RAW (rms)'
    for thruster in thruster_dict.keys():
        all_files+=thruster_dict[thruster][sensor]

    for file in all_files:
        label = file.strip('.csv').split('/')[-1].split('_')[-1]
        if label[faultkey] == '1':
            faultyfilepaths.append(file)
        else:
            healthyfilepaths.append(file)

    if len(healthyfilepaths) < len(faultyfilepaths):
        max_num_samples = len(healthyfilepaths)
    else:
        max_num_samples = len(faultyfilepaths)

    healthyfilepaths = np.random.choice(healthyfilepaths, size = max_num_samples, replace = False).tolist()
    faultyfilepaths = np.random.choice(faultyfilepaths, size = max_num_samples, replace = False).tolist()

    healthy_trainset, healthy_valset = train_test_split(healthyfilepaths, test_size = 0.33,shuffle = True)

    faulty_trainset, faulty_valset = train_test_split(faultyfilepaths, test_size = 0.33,shuffle = True)


    all_testfilepaths+=thruster_dict[test_thruster][test_sensor]

    for file in all_testfilepaths:
        label = file.strip('.csv').split('/')[-1].split('_')[-1]
        if label[faultkey]=='1':
            faulty_testfilepaths.append(file)
        else:
            healthy_testfilepaths.append(file)
    #print("len(healthy_testfilepaths)",len(healthy_testfilepaths))
    #print("len(faulty_testfilepaths)",len(faulty_testfilepaths))

    if len(healthy_testfilepaths) < len(faulty_testfilepaths):
        max_num_testsamples = len(healthy_testfilepaths)
    else:
        max_num_testsamples = len(faulty_testfilepaths)

    healthy_testfilepaths = np.random.choice(healthy_testfilepaths, size = max_num_testsamples, replace = False).tolist()
    faulty_testfilepaths = np.random.choice(faulty_testfilepaths, size = max_num_testsamples, replace = False).tolist()


    trainset = healthy_trainset+faulty_trainset
    valset = healthy_valset+faulty_valset
    testset = healthy_testfilepaths+faulty_testfilepaths
    return trainset, valset, testset

def get_sensor_separated_thruster_filepaths(faults,vessel_dir):
    """Randomly divides data measured with Input shaft RAW sensor from thrusters 1,5 and 6
    in to train and validation datasets. These datasets hold 50 % of faulty data
    and 50 % of healthy data.
    Each fault has its own train, validation and test dataset.
    Test datasets for each fault are measured with different sensors

    In detail, the following testsets are constructed with this function:

    - Inner race defect:
        - Source: Thruster 1 | Pinion shaft gear end RAW (sensor)
    - Gear wheel defect:
        - Source: Thruster 6 | Propeller shaft fwd end RAW (sensor)

    IN:
    dict - fault dictionary {'fault type' : 'label index'}

    OUT:
    dict - {'fault_type':{'train':[*filepaths], 'val':[*filepaths], 'test': [*filepaths]}}
    """
    thruster_filepaths = {}
    thruster_dict = get_thruster_files(vessel_dir)

    for fault, index in faults.items():
        #print(fault, index)
        thruster_filepaths[fault] = {}
        trainsplit,valsplit, testsplit = get_splits_sensor_thruster(index, thruster_dict)
        thruster_filepaths[fault]['train'] = trainsplit
        thruster_filepaths[fault]['val'] = valsplit
        thruster_filepaths[fault]['test'] = testsplit

    return thruster_filepaths

def save_thruster_history(history,SAVEPATH,model_name,fault_type):
    fig,axes = plt.subplots()
    l1, = axes.plot(history['train'],label = "Training error")
    l2, = axes.plot(history['val'], label = "Validation error")
    plt.legend(handles=[l1,l2])
    axes.set_title('Convergence of '+model_name)
    axes.set_ylabel('CrossEntropyLoss')
    axes.set_xlabel('Epoch')
    fig.savefig(SAVEPATH+model_name+'-'+fault_type+'.png')
    plt.close(fig)

if __name__ == '__main__':
    faults = {'bearing': 4,
            'pinion': 8,
            'wheel': 9}
    vessel_dir = '../original.tmp/labeled_data'
    get_sensor_separated_thruster_filepaths(faults,vessel_dir)
    #get_spearated_thruster_filepaths(faults,vessel_dir, train = ['1','6'], test = ['5'])

    #None = get_random_thruster_filepaths(faults,vessel_dir)
    #get_thruster_files('../original.tmp/labeled_data')
