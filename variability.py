import json
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
from operator import itemgetter 


# with open('data.json') as f:
#     data = json.load(f)

class Labels():
    def __init__(self, data, TIMESTEP_THRESHOLD):
        """
        This Labels class takes in data, and a Timestep threshold, and in the main method
        generates a list of labels the size of data so that periods with high Heart Rate
        Variability (HRV) above the threshold are labeled 1, and those below the threshold
        are labeled 0.
        """
        super(Labels, self).__init__()
        self.data = data
        self.TIMESTEP_THRESHOLD = TIMESTEP_THRESHOLD

    '''
    Takes in BVP data as a list and returns list of indices of local maxima
    '''
    def gen_local_maxima(self, data):
        return find_peaks(data)

    '''
    Takes in indices of maxima and calculates the timestep between them

    :returns - nn_list, where each index in list, i, represents the timestep (measured in seconds)
                between the the current maxima and the next
    '''
    def find_timestep_differences(self, indices):
        # print(indices)
        nn_list = [indices[i+1]-indices[i] for i,elt in enumerate(indices) if i!=len(indices)-1]
        #include 0 at end since last maxima doesn't have a timestep
        nn_list.append(0)
        # print(nn_list)
        return nn_list


    '''
    :input => list of timesteps between peaks in BVP data
    :return => list of binary labels of the size of our original BVP data, where
                the label is 1 AT AND BETWEEN PEAKS if the corresponding timestep between optima is
                greater than our TIMESTEP_THRESHOLD. Else the label between peaks is 0
    '''
    def calc_stress_hrv(self, timesteps, indices, data):
        stressful_peaks = []
        for timestep in timesteps:
            if timestep > self.TIMESTEP_THRESHOLD:
                stressful_peaks.append(1)
            else:
                stressful_peaks.append(0)
        
        # print(stressful_peaks)
        
        #now take binary labels and fill out original list the size of original BVP data
        labels = [0] * len(data)
        for i, index in enumerate(indices):
            labels[index] = 1
        # print(labels) #labels is now size of data with 1 at local optima:

        #go through and fill in gaps in labels depending on whether the period is stressful or not
        new_labels= labels.copy()
        BINARY_VAR = 0
        for i, obs in enumerate(labels):
            if obs == 1:
                timestep_index = indices.index(i)
                if stressful_peaks[timestep_index] == 1:
                    BINARY_VAR = 1
                else:
                    BINARY_VAR = 0
            
            new_labels[i] = BINARY_VAR

        return new_labels

    '''
    Runs through helper methods and generates new labels based on HRV 
    for particular patient's time series BVP data:
    '''
    def main(self):
        maxima_indices = self.gen_local_maxima(self.data)[0].tolist()
        timesteps = self.find_timestep_differences(maxima_indices)
        new_labels = self.calc_stress_hrv(timesteps, maxima_indices, self.data)
        return new_labels

    # plt.plot(new_labels)
    # plt.show()

