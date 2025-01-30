#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 09:41:53 2025

@author: nbonfiglio
"""

import mne

import os 

data_path = '/export/home/nbonfiglio/public/EEL-Hack/bids'
path_eelbrain = '/export/home/nbonfiglio/public/EEL-Hack/eelbrain/eeg'

names=[]
for fname in os.listdir(data_path):
    if "sub-" in fname:
        names.append(fname)

# bad subjects
names.remove('sub-25')
names.remove('sub-29')

veog="VeogL"
heog='HeogL'

# dictionary of manually selected components
ica_exclude_manual = {'sub-1' : [0, 5, 8], 'sub-2' : [0, 2], 'sub-3' : [0, 1], 'sub-6' : [0, 10], 'sub-9' : [0, 9], 'sub-10' : [2, 11],
                       'sub-11': [0, 2, 3], 'sub-12': [0, 5], 'sub-13': [0, 3], 'sub-15': [1, 5, 9], 'sub-16': [0, 2], 'sub-17':[0, 13],
                       'sub-20': [0, 6], 'sub-22': [0, 5], 'sub-24' : [0, 3], 'sub-26' : [0, 2], 'sub-27': [0, 4, 12], 'sub-30' : [1, 3],
                       'sub-31': [0, 2], 'sub-32' : [0, 3], 'sub-33': [0, 5]}

for sub in names:

    raw = mne.io.read_raw_brainvision(os.path.join(data_path, sub, 'eeg', sub + '_task-entrainment_run-01_eeg.vhdr'), preload= True)
    
    # set relevant information
    raw.set_channel_types(mapping={'VeogL': 'eog', 'HeogR': 'eog', 'HeogL':'eog', 'A1':'emg', 'A2':'emg', 'Audio':'misc'})
    # since there is no channel type for mastoid reference, we put emg not to raise incompatibilities with the montage
    raw.set_eeg_reference(ref_channels=['A1', 'A2'])
    raw.set_montage('easycap-M1')

    # filter
    raw_filtered = raw.copy().filter( 
    l_freq = 0.01, # remove very slow drifts
    h_freq = 35,  # We follow the original paper with low-pass filter at 35 Hz
    picks = 'eeg',  # Channels to include
    phase = 'zero-double',  # Compensates the delay of the filter, and applies it two times: forwards and backwards.
    n_jobs = 6,  # Number of jobs to run in parallel; -1 sets it to the number of CPU cores.
    verbose = False)
    
    path_ica_solution = os.path.join(data_path, 'derivatives', sub, 'ica_plots','ica_solution.fif')
    ica = mne.preprocessing.read_ica(path_ica_solution)
    
    if sub in ica_exclude_manual:
        
        reject_icas = ica_exclude_manual[sub]     
    
    else: # if not in the list of manually selected, automatic selection is ok
        
        raw_forICA = raw.copy().filter(l_freq = 0.5, h_freq = 30, verbose = False)
        veog_indices, veog_scores = ica.find_bads_eog(raw_forICA, ch_name=veog, threshold=2.5)
        heog_indices, heog_scores = ica.find_bads_eog(raw_forICA, ch_name=heog, threshold=2.5)
    
        reject_icas= veog_indices + heog_indices 
    
    raw_filtered_ica=ica.apply(raw_filtered, exclude=reject_icas)
    
    sub_number = 'sub-'+ sub[4:].zfill(2)
    
    os.makedirs(os.path.join(path_eelbrain, sub_number)) 
    # eelbrain requires one folder for each subject
    raw_filtered_ica.save(os.path.join(path_eelbrain, sub_number, sub_number + '_story-raw.fif'))
    # this is the naming convention of eelbrain
    
       