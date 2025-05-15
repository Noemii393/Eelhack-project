#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 15:27:28 2024

@author: nbonfiglio
"""
import os
import os.path as op

import numpy as np

import mne
from mne_bids import (
    write_raw_bids,
    read_raw_bids,
    BIDSPath,
    mark_channels)

import eelbrain

import wavfile # to read the wav 

from find_delay import find_delay  
# this can be freely installed with: pip install find-delay
# we recommend checking the documentation and other examples at: https://find-delay.readthedocs.io/en/latest/#

# paths
path = '/bcbl/home/public/EEL-Hack/bids/derivatives'
path_epochs ='/bcbl/home/public/EEL-Hack/Scripts'
path_audio = "/bcbl/home/public/EEL-Hack/bids/stimuli"

# load data
epochs = eelbrain.load.unpickle(path_epochs + '/epochs_sub13.pickle') # load the epochs to get the time start of the audio for this subject   

path_bids = BIDSPath(subject = '13', task = 'entrainment', run = '01', suffix = "eeg", processing = 'ica', root = path)
raw = read_raw_bids(path_bids)

# load the audio
wav = wavfile.read(path_audio + '/stimulus.wav')
wav_freq = wav[1] # sampling frequency  
wav_array = wav[0]

# Get duration and onset/offset time points in the raw file
duration = len(wav_array) / wav_freq

onset = epochs['i_start'][0] # in samples 
offset = onset + duration * raw.info["sfreq"] # transform in sample units    

# In meg systems, the first sample is not 0, so we should subtract that value, stored in raw as "first_samp"
'''abs_onset = onset - raw.first_samp
abs_offset = offset - raw.first_samp'''

# Create your audio array from the Audio channel of the eeg recording
eeg_audio_array = raw.pick(["Audio"])[0, onset:offset][0]
eeg_audio_array = np.squeeze(eeg_audio_array) # it must be 1-dimensional    

wav_array = np.squeeze(wav_array) # it must be 1-dimensional

delay = find_delay(array_1 = eeg_audio_array,
                   array_2 = wav_array,
                   freq_array_1 = raw.info["sfreq"],
                   freq_array_2 = wav_freq,
                   resampling_rate = min(raw.info["sfreq"], wav_freq),
                   return_delay_format = "ms",
                   threshold = 0.1, plot_figure = True, # we will check the delay by inspecting the figure  
                   path_figure=path_epochs + "/figure_delay_13.png",
                   name_array_1="Audio_eeg", name_array_2="Audio_wav",
                   remove_average_array_1 = True) # useful in case the audio channel is not centered at 0   