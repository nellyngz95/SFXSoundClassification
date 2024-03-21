import urllib
import torchaudio
import glob
import torch
import os
import numpy as np  
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
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
    