import torch
import urllib
import torchaudio
import glob
import os   
import numpy as np  
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from VGGSoundClassification.Data import NDataset, NDataLoader
#Downloads the vggish model from harritaylor's github and the checkpoint file
#model = torch.hub.load('harritaylor/torchvggish', 'vggish')
model = VGGISH.get_model().to(device)


model.eval()

def vggish_melspectrogram(audio_path):
    melspec_proc = VGGISH.get_input_processor()
    waveform, original_rate = torchaudio.load(audio_path)
    waveform = waveform.squeeze(0)
    waveform = torchaudio.functional.resample(waveform, original_rate, VGGISH.sample_rate)
    melspec = melspec_proc(waveform)
    return melspec

#Extracts the features from the audio file
#ypou can replace the filename with the path to your own audio file
file_dir= '/Users/nellygarcia/Documents/InformationRetrivalPhd/Dataset'
embeddings = '/Users/nellygarcia/Documents/GitHub/PythonNelly/VGGSoundClassification/embeddings'
if not os.path.exists(embeddings):
    os.mkdir(embeddings)    

for  index, file_path in enumerate (glob.glob(f"{file_dir}/**/*.wav", recursive=True)):
    #waveform, sample_rate = torchaudio.load(file_path)
    melspec = vggish_melspectrogram(file_path)
    features = model(melspec)
    
    embedding = features.detach()
    fname = os.path.basename(file_path).split('.')[0]
    torch.save(embedding, f"{embeddings}/{fname}.pt")
    #numpy() #convert them to an array ?? The kernel crashes when I try to convert them to a tensor  
#
print("total files processed: ", index+1)
    
