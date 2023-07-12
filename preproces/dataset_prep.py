import os

import numpy as np
from scipy.io import loadmat
from sklearn.utils import shuffle


def create_dataset(elec_idx,data_length,folder_path,trials_num=6,classes_num=40,sub_num=35):
    '''
	loads .mat files (64 x 1500 x 40 x 6) with EEG data and combines them into one dataset

        Parameters:
	elec_idx: index of electrodes data to keep
	data_length: length of data from each trial to use for classification [in number of samples]: max - 1500
	folder_path: path to the folder with .mat files
	trials_num: number of trials: max - 6
	classes_num: number of classes:  max - 40
	sub_num: number of subjects: max - 35

	returns: np.array: samples x eeg_channels x time_points
    '''

    num_of_elec = len(elec_idx)
    X = np.array([],dtype=np.float32).reshape(0,len(elec_idx),data_length)
    samples_per_sub = trials_num*classes_num
      
    for file_name in os.scandir(folder_path): #iterates over files in the folder_path 
        file_content = loadmat(file_name.path)['data'][elec_idx,125:data_length+125,:,:] 
        file_content = file_content.reshape(num_of_elec,data_length,samples_per_sub).transpose(2,0,1) # [samples,channels,timepoints]
        X = np.concatenate([X,file_content],axis=0)
    y = np.repeat(np.array([np.repeat(i,trials_num) for i in range(classes_num)],dtype=np.int32).reshape(1,samples_per_sub),sub_num,axis=0).reshape(samples_per_sub*sub_num) # generates sparse labels
    group_size = int(X.shape[0]/sub_num)
    for i in range(sub_num): #shuffles data from each person separately
        X[group_size*i:group_size*(i+1)],y[group_size*i:group_size*(i+1)] = shuffle(X[group_size*i:group_size*(i+1)],y[group_size*i:group_size*(i+1)])
    return X,y


def split_data(X,y, length,split,classes_num,sub_num,trials_num):
    """
    in case of subject-dependent classification splits the data into training, validation, and testing sets.

    Parameters:
    X (ndarray): data to split
    y (ndarray): labels to split
    length (int): The length of the data.
    split (float): The proportion of data to allocate for training. Value should be between 0 and 1.
    classes_num: number of classes
    sub_num: number of subjects
    truals_num: number of trials

    Returns:
    x_train (ndarray): The training data with shape (N_train, channels, length).
    x_val (ndarray): The validation data with shape (N_val, channels, length).
    x_test (ndarray): The testing data with shape (N_test, channels, length).
    y_train (ndarray): The labels for the training data with shape (N_train,).
    y_val (ndarray): The labels for the validation data with shape (N_val,).
    y_test (ndarray): The labels for the testing data with shape (N_test,).
    """

    samples_per_sub = data.shape[0]/sub_num
    part = split*samples_per_sub

    x_train = np.zeros((int(classes_num*trials_num*split*sub_num), data.shape[1],length))
    y_train = np.zeros(int(classes_num*trials_num*split*sub_num)).astype(np.int32)
    x_val = np.zeros((int(classes_num*trials_num*round((1-split)/2,1)*sub_num),data.shape[1], length))
    y_val = np.zeros(int(classes_num*trials_num*round((1-split)/2,1)*sub_num)).astype(np.int32)
    x_test = np.zeros((int(classes_num*trials_num*round((1-split)/2,1)*sub_num),data.shape[1], length))
    y_test = np.zeros(int(classes_num*trials_num*round((1-split)/2,1)*sub_num)).astype(np.int32)
    
    for i in range(sub_num):
        x_train[int(part*i):int(part*(i+1))] = X[int(part*i+(samples_per_sub-part)*i):int(part*(i+1)+(samples_per_sub-part)*i)]
        y_train[int(part*i):int(part*(i+1))] = y[int(part*i+(sampleS_per_sub-part)*i):int(part*(i+1)+(samples_per_sub-part)*i)]
        x_val[int((samples_per_sub-part)/2)*i:int((samples_per_sub-part)/2)*(i+1)] = X[int(part*(i+1)+(samples_per_sub-part)*i):int(samples_per_sub*(i+1)-int((samples_per_sub-part)/2))]
        y_val[int((samples_per_sub-part)/2)*i:int((samples_per_sub-part)/2)*(i+1)] = y[int(part*(i+1)+(samples_per_sub-part)*i):int(samples_per_sub*(i+1))-int((samples_per_sub-part)/2)]
        x_test[int((samples_per_sub-part)/2)*i:int((samples_per_sub-part)/2)*(i+1)] = X[int(part*(i+1)+(samples_per_sub-part)*i)+int((samples_per_sub-part)/2):int(samples_per_sub*(i+1))]
        y_test[int((samples_per_sub-part)/2)*i:int((samples_per_sub-part)/2)*(i+1)] = y[int(part*(i+1)+(samples_per_sub-part)*i)+int((samples_per_sub-part)/2):int(samples_per_sub*(i+1))]

    return x_train,x_val,x_test,y_train,y_val,y_test



