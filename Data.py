import torch
import torchaudio
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import pandas as pd
import os
import json
import random
import pytorch_lightning as pl
import torch.nn as nn   
import torch.nn.functional as F




# Creating the Dataset.
class NDataset(torch.utils.data.Dataset):
    def __init__(self, 
                 annotation_file_dir: str, 
                 embed_dir: str, 
                 division:str = "train",
                 n_frames = 5,
                 ): 
        super().__init__()
        #load the annotations
        if division not in ["train", "test", "validation"]:
            raise Exception("Division not supported")
        self.annotations = pd.read_csv(os.path.join(annotation_file_dir, f"{division}.csv"))
        print(f"found {len(self.annotations)} samples")

        

        self.embed_dir = embed_dir #Embeddings directory .pt
       # self.annotations=pd.read_csv(annotation_file_dir) #Annotations directory .csv
        #Check number of categories.
        self.num_categories = len(self.annotations.iloc[:,2].unique())
        self.n_frames = n_frames
        print(f"found {self.num_categories} categories")

    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, index):

        #get the file name and label from the annotations
        embedding_file = self.annotations.iloc[index,1]
        ##print (audio_file)
        one_hot_label = self.annotations.iloc[index,3]
        #covert string to tensor
        one_hot_label = torch.tensor(json.loads(one_hot_label)).float()
        #print(one_hot_label)
        #print(type(one_hot_label))
        self.ignore_idx = []

        #load the audio file
        try:
            emb = torch.load(embedding_file)
        except:
            self.ignore_idx.append(index)
            return self[index + 1]
        #print(signal.shape)
        #pad the audio file if necessary
        emb = self._pad_if_necesary(emb)
       
        return emb, one_hot_label
    
      

    def _pad_if_necesary(self, emb):
        # Pad time axis to n_frames
        if emb.shape[0] < self.n_frames:
            # pad first dimension (time) to n_frames
            emb = F.pad(emb, (0, 0, 0, self.n_frames - emb.shape[0]))
        return emb

    def _cut_if_necessary(self, signal):
        if signal.shape[1] > self.num_samples:
            signal = signal[:, :self.num_samples]
        return signal


