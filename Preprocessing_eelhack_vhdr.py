#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 13:07:53 2024

@author: nbonfiglio
"""
import os
import os.path as op

# Perform plots
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np

# MEG processing
import mne
from mne_bids import (
    write_raw_bids,
    read_raw_bids,
    BIDSPath,
    mark_channels)

# Configuration
path_data='/export/home/nbonfiglio/lab/EEG/NEUROPARENTS/DATA/EEG/raw_data/'
path_output = '/export/home/nbonfiglio/public/Exchange/NoemiVincenzo/bids'
path_derivatives='/export/home/nbonfiglio/public/Exchange/NoemiVincenzo/bids/derivatives'

names=[]
for fname in os.listdir(path_data):
    if "ent_esp.vhdr" in fname:
        names.append(fname)
  
# first writing bids
for i in range(len(names)):
    raw=mne.io.read_raw_brainvision(path_data + '/' + names[i], ignore_marker_types=True)

    path_bids_raw = BIDSPath(subject = str(i+1), task = 'entrainment', run = '01', suffix = "eeg", root = path_output)
    write_raw_bids(raw = raw, bids_path = path_bids_raw, overwrite = True, verbose = False)
    
    # set output paths for preprocessing
    path_bids_deriv=BIDSPath(subject = str(i+1), task = 'entrainment', run = '01', processing='ica', suffix = "eeg", root = path_derivatives)   
    os.makedirs(path_derivatives + '/sub-' + str(i+1) + '/ica_plots')
    path_ica=path_derivatives + '/sub-' + str(i+1) + '/ica_plots'
    
    # now load the data for preprocessing 
    raw=mne.io.read_raw_brainvision(path_data + '/' + names[i], ignore_marker_types=True, preload = True)
    
    # set relevant information (note: for the features of the vhdr format, channel types and montage need be specified again when reading with eelbrain)
    raw.set_channel_types(mapping={'VeogL': 'eog', 'HeogR': 'eog', 'HeogL':'eog', 'A1':'emg', 'A2':'emg', 'Audio':'misc'})
    raw.set_eeg_reference(ref_channels=['A1', 'A2'])
    raw.set_montage('easycap-M1')
    
    # filter and ica
    raw_filtered = raw.load_data().copy().filter( 
        l_freq = 0.01, # remove very slow drifts
        h_freq = 35,  # We follow the original paper with low-pass filter at 35 Hz
        picks = 'eeg',  # Channels to include
        phase = 'zero-double',  # Compensates the delay of the filter, and applies it two times: forwards and backwards.
        n_jobs = 6,  # Number of jobs to run in parallel; -1 sets it to the number of CPU cores.
        verbose = False)
    
    raw_forICA = raw.copy().filter(l_freq = 0.5, h_freq = 30, verbose = False) # another version with stricter filter, just to compute ica components   

    ica = mne.preprocessing.ICA(n_components = 20, random_state=97).fit(raw_forICA, picks='eeg')
    
    ica.save(path_ica + '/ica_solution.fif')
    
    # detect and reject eog components
    veog="VeogL"
    heog='HeogL'

    veog_indices, veog_scores = ica.find_bads_eog(raw_forICA, ch_name=veog, threshold=2.5)
    heog_indices, heog_scores = ica.find_bads_eog(raw_forICA, ch_name=heog, threshold=2.5)

    reject_icas= veog_indices + heog_indices 
    
    if reject_icas: # if the list is not empty, otherwise it would raise error
        # apply and save
        raw_filtered_ica=ica.apply(raw_filtered, exclude=reject_icas)
        write_raw_bids(raw = raw_filtered_ica, bids_path = path_bids_deriv, format='BrainVision', overwrite = True, verbose = False, allow_preload=True)
        # save for later inspection
        f5=ica.plot_components(picks=reject_icas, show=False)
        plt.savefig(path_ica + '/ICA_reject_components_topo.png', dpi=300)

    #save scores
    f1=ica.plot_scores(veog_scores, show=False)
    plt.savefig(path_ica +'/veog_scores.png')
    
    f2=ica.plot_scores(heog_scores, show=False)
    plt.savefig(path_ica + '/heog_scores.png', dpi=300)
    
    # save topoplots of all components for later inspection
    f4=ica.plot_components(sphere='auto', show=False)
    plt.savefig(path_ica+'/ICA_all_components_topo.png')
        
    plt.close("all")
    
# now exclude the right components, after visual inspection 

# load participants names
names=[]
for fname in os.listdir(path_derivatives):
    if "sub-" in fname:
        names.append(fname)

# create a dictionary for components to exclude, manual selection according to the plots (missing participants were ok with automatic rejection)
ica_exclude = {'sub-1' : [0, 5, 8], 'sub-2' : [0, 2], 'sub-3' : [0, 1], 'sub-6' : [0, 10], 'sub-9' : [0, 9], 'sub-10' : [2, 11],
               'sub-11': [0, 2, 3], 'sub-12': [0, 5], 'sub-13': [0, 3], 'sub-15': [1, 5, 9], 'sub-16': [0, 2], 'sub-17':[0, 13],
               'sub-20': [0, 6], 'sub-22': [0, 5], 'sub-24' : [0, 3], 'sub-26' : [0, 2], 'sub-27': [0, 4, 12], 'sub-30' : [1, 3],
               'sub-31': [0, 2], 'sub-32' : [0, 3], 'sub-33': [0, 5]}
        
for sub in names:
    if sub in ica_exclude:
        path_bids_raw = BIDSPath(subject = sub[4:], task = 'entrainment', run = '01', suffix = "eeg", root = path_output)
        raw = read_raw_bids(path_bids_raw) # load the raw, without filter and ica  
        
        raw.load_data()
    
        # set relevant information 
        raw.set_channel_types(mapping={'VeogL': 'eog', 'HeogR': 'eog', 'HeogL':'eog', 'A1':'emg', 'A2':'emg', 'Audio':'misc'})
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
        
        # load ica solution 
        path_ica_solution = path_derivatives + '/' + sub + '/ica_plots/ica_solution.fif'
        ica = mne.preprocessing.read_ica(path_ica_solution)
        
        # define components and exlude 
        reject_icas = ica_exclude[sub]     
        
        raw_filtered_ica=ica.apply(raw_filtered, exclude=reject_icas)
        
        # save bids
        path_bids_deriv = BIDSPath(subject = sub[4:], task = 'entrainment', run = '01', processing='ica', suffix = "eeg", root = path_derivatives)   
        
        write_raw_bids(raw = raw_filtered_ica, bids_path = path_bids_deriv, format='BrainVision', overwrite = True, verbose = False, allow_preload=True)
        
        # save new plot
        f5=ica.plot_components(picks=reject_icas, show=False)
        path_ica=path_derivatives + '/' + sub + '/ica_plots'
        plt.savefig(path_ica + '/ICA_reject_components_topo_second.png', dpi=300)
        
        plt.close("all")
    
        
        
        
            



















































































































    
    
    
    
    
  