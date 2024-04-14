import urllib
import torchaudio
import glob
import torch
import os
import numpy as np  
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torch import hub
import sys
sys.path.append('/Users/nellygarcia/Documents/SFXSoundClassification')
import vggish_input, vggish_params
#Building a Classifier
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(640, 256)
        self.fc3 = nn.Linear(256, 32)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
       # x = self.dropout(x)
       # x = F.relu(self.fc2(x))
       # x = self.dropout(x)
        x = F.log_softmax(self.fc3(x), dim=1)
        return x
    

class VGG(nn.Module):
    def __init__(self, features):
        super(VGG, self).__init__()
        self.features = features
        self.embeddings = nn.Sequential(
            nn.Linear(512 * 4 * 6, 4096),
            nn.ReLU(True),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Linear(4096, 128),
            nn.ReLU(True))

    def forward(self, x):
        x = self.features(x)

        # Transpose the output from features to
        # remain compatible with vggish embeddings
        x = torch.transpose(x, 1, 3)
        x = torch.transpose(x, 1, 2)
        x = x.contiguous()
        x = x.view(x.size(0), -1)

        return self.embeddings(x)
