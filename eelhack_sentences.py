#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 15:04:35 2024

@author: nbonfiglio
"""
# This file contains a pipeline specification. The pipeline is defined as a subclass of TRFExperiment, and adds information about the specific paradigm and data.
from eelbrain.pipeline import *
from trftools.pipeline import *

import mne
import pandas as pd

# This is the root directory where the pipeline expects the experiment's data
# in the format specified by eelbrain (NO BIDS - see documentation) 
DATA_ROOT = "/bcbl/home/public/EEL-Hack/eelbrain"

# first let's load the events to define the duration of our trials
events = pd.read_csv("/bcbl/home/public/EEL-Hack/bids/derivatives/sub-1/eeg/sub-1_task-entrainment_run-01_proc-ica_events.tsv", sep = '\t')
# to do: find a way to code this for every subject

events_sentences = events[events['value'].isin (range(2,92))] # select only the events related to sentences
events_sentences["duration"] = events_sentences ["onset"].diff().shift(-1) # calculate durations
SEGMENT_DURATION = dict(zip(range(2, 91), events_sentences["duration"])) # save them in a dictionary, ignoring last sentence for now as it doesn't have a  following trigger

# One may also want to define parameters used for estimating TRFs here, as they are often re-used along with the pipeline in multiple location 
PARAMETERS = {
    'raw': '0.5-20', # the data that we want to use (e.g., filtered between 0.5-20 Hz)
    'samplingrate': 50, # downsampling for efficiency
    'data': 'eeg', # type of data
    'tstart': -0.100,
    'tstop': 0.600, # time range for fitting (training) the TRF
    'filter_x': 'continuous', # filtering continous predictors the same way as the eeg
    'error': 'l1',
    'basis': 0.050,
    'partitions': -5,
    'selective_stopping': 1, # whether to stop training for one predictors when it's reaching overfitting (1=True)
}


# Start by defining a subclass of the pipeline
class Eelhack(TRFExperiment):

    # This is the directory withing the root directory that contains the EEG data
    data_dir = 'eeg'
    # This is how subject names are identified ("sub" followed by two digits). See the documentation of the builtin Python regular expression (re) module for details on how to build patterns
    subject_re = r'sub-\d\d'

    # This is used to identify the *-raw.fif file in the eeg directory 
    sessions = ['story']

    # This defines the preprocessing pipeline. In  our case, the "RawSource" file is already after ICA
    raw = {
        'raw': RawSource(filename='{subject}_{recording}-raw.vhdr', connectivity='auto', reader=mne.io.read_raw_brainvision, eog=['VeogL', 'HeogR', 'HeogL', 'A1','A2'], montage='easycap-M1'),
        # here we specify 1) that the file must be read with the function for brainvision format (default would be fif), 2) that those 5 channels are not eeg, and 3) the montage that was used
        '0.5-20': RawFilter('raw', 0.5, 20, cache=False) # additional filter for TRF
    }

    # This defines our stimulus of interest (the sentences) and their duration (from the dictionary above)
    variables = {
        'stimulus': LabelVar('trigger', {tuple(range(2,92)): 'sentence'}),
        'duration': LabelVar ('trigger', SEGMENT_DURATION)       
    }

    # This defines the epochs using the triggers and duration specified above 
    epochs = {
        'sentences': PrimaryEpoch('story', "stimulus.isin('sentence')", tmin=0, tmax='duration', samplingrate=50)
    }

    # This defines which variable (among the variables assigned to the events in the EEG recordings) designates the stimulus. Eelbrain will select the corresponding predictor. 
    stim_var = 'trigger' # in our case, we use "trigger" because it is the variable we use to epoch and to create the matching predictors (see the "make_predictors_sentences" script) 
    
    # This specifies how the pipeline will find and handle predictors. The key indicates the first part of the filename in the <root>/predictors directory. 
    predictors = {
        'gammatone': FilePredictor(resample='bin'),
        'gammatonon': FilePredictor(resample='bin') # we call the onset predictors "gammatonon" (insted of 'gammatone-on') to avoid conflict with the "-" symbol
    }
    
    # Models are shortcuts for invoking multiple predictors
    models = {
        'auditory-envelope': 'gammatone-1 + gammatonon-1', # envelope + acoustic onsets. '-1' indicates the one-band version of the gammatone (i.e., the envelope) 
    }

# This creates an instance of the pipeline. Doing this here will allow other scripts to import the instance directly.
eelhack=Eelhack(DATA_ROOT)

