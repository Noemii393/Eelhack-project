"""Generate high-resolution gammatone spectrograms"""
from pathlib import Path

import eelbrain
import wave

path = Path('/bcbl/home/public/EEL-Hack/bids/stimuli')# path for the stimuli (in this case a single audio)
output_path =  Path('/bcbl/home/public/EEL-Hack/eelbrain/predictors') # Note: eelbrain requires a folder named 'predictors'  
wav = wave.open(str(path / 'stimulus.wav'))

epochs = eelbrain.load.unpickle(path / 'epochs.pickle') # load the epochs as defined within the script 'makeTRF' to create predictors that match them 

with wave.open(str(path / 'stimulus.wav'), 'rb') as infile:
    # get file data
    nchannels = infile.getnchannels()
    sampwidth = infile.getsampwidth()
    framerate = infile.getframerate()
    # segment audio for each epoch
    for i in range(len(epochs['T'])):
        start = epochs['T'][i] # starts at time "T" of the epoch
        infile.setpos(int( (start - epochs['T'][0]) * framerate)) # define the starting position to read the audio
        # we subtract the first trigger's timestamp (i.e. the delay between the start of the eeg recording and the start of the audio), and transform in sample 
        
        if i + 1 == len(epochs['T']): # if you reach the last epoch, get the audio until the end
            end = infile.getnframes()/framerate
        else:        
            end = epochs['T'][i+1] # if not, define the end as the "T" of the following epoch      
        
        # extract data with the desired duration from the starting position above 
        data = infile.readframes(int((end - start) * framerate))
        print(start, end)
        
        # write the extracted data to a new file called as the trigger of the epoch
        with wave.open(str(path /f'{epochs["trigger"][i]}.wav'), 'w') as outfile:
            outfile.setnchannels(nchannels)
            outfile.setsampwidth(sampwidth)
            outfile.setframerate(framerate)
            outfile.setnframes(int(len(data) * sampwidth))
            outfile.writeframes(data)

# now that we have one audio for each epoch, we can create predictors
for i in range(len(epochs['T'])):
  
    wav = eelbrain.load.wav(str(path /f'{epochs["trigger"][i]}.wav'))
                        
    gammatone = eelbrain.gammatone_bank(wav, 80, 15000, 128, location='left', tstep=0.001) # spectogram 
    #Apply a log transform to approximate peripheral auditory processing
    gt_log = (gammatone + 1).log()
    # Apply the edge detector model to generate an acoustic onset spectrogram
    gt_on = eelbrain.edge_detector(gt_log, c=30)
    
    # Create and save 1 band versions of the two predictors (i.e., temporal envelope predictors, made by the sum of frequencies)
    eelbrain.save.pickle(gt_log.sum('frequency'), output_path / f'{epochs["trigger"][i]}~gammatone-1.pickle')
    eelbrain.save.pickle(gt_on.sum('frequency'), output_path / f'{epochs["trigger"][i]}~gammatonon-1.pickle')
                       
    gammatone_name = str(output_path / f'{epochs["trigger"][i]}-gammatone.pickle') # again, call the output as the trigger of the epoch
    eelbrain.save.pickle(gammatone, gammatone_name)