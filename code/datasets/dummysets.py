import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset # This is the standard dataset class

class DummyClasses(Dataset):
    def __init__(self, labels = [1,2,3], n_samples = 200, input_size = 2048):
        """
        This class initialises n_samples for each class in labels.
        One sample is 2048x1 sample of noise drawn from gaussian distribution.
        The mean of the gaussian distribution is pointed by the label. The
        variance is 1.
        """

        self.samples = generate_samples(labels,n_samples, input_size)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self,i):
        return self.samples[i]


def generate_samples(labels,n_samples, input_size):
    """
    This function generates n_samples amount of samples for every label in
    labels.
    """
    samples = []
    var = 1

    for mean in labels:
        values = np.random.normal(loc= mean, scale = var,size = (n_samples, input_size))
        print(values.shape)
        for i in range(values.shape[0]):
            samples.append((torch.Tensor(values[i]).unsqueeze(0),mean))
    return samples

if __name__ == '__main__':
    generate_samples(labels = [1,2,3], n_samples = 200)
