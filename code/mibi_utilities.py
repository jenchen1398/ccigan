import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils
from torch.nn.utils import spectral_norm
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
from torch.utils.data import DataLoader
from ds import MIBIDataset

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.xavier_uniform(m.weight.data)
        
# input size seg: (6 x 64 x 64)

def seg_show(seg):
    n_class = 17
    colors = []
    
    # Gray: Unidentified
    Gray = np.zeros([64,64,3])
    Gray[:,:,:] = 0.8
    colors.append(Gray)
    
    # Purple: Endothelial
    Purple = np.zeros([64,64,3])
    Purple[:,:,0] = 1
    Purple[:,:,2] = 1
    colors.append(Purple)
    
    # Cyan: Mesenchymal-like
    C = np.zeros([64,64,3])
    C[:,:,0] = 1
    C[:,:,1] = 0.5
    C[:,:,2] = 0.6
    colors.append(C)
    
    # Yellow: 5 Tumor
    Yellow = np.zeros([64,64,3])
    Yellow[:,:,0] = 1
    Yellow[:,:,1] = 1
    colors.append(Yellow)
    
    # Red: 6 Keratin-Positive Tumor
    Red = np.zeros([64,64,3])
    Red[:,:,0] = 1
    colors.append(Red)
    
    
    # Immune:
    Cyan = np.zeros([64,64,3])
    Cyan[:,:,1] = 1
    Cyan[:,:,2] = 1
    colors.append(Cyan)
    
    
    # CD4
    Imunne = np.zeros([64,64,3])
    Imunne[:,:,0] = 1.0
    Imunne[:,:,1] = 0.8
    Imunne[:,:,2] = 0.0
    colors.append(Imunne)
    
    # CD8
    Imunne = np.zeros([64,64,3])
    Imunne[:,:,0] = 1.0
    Imunne[:,:,1] = 0.6
    Imunne[:,:,2] = 0.0
    colors.append(Imunne)
    
    # CD3
    Imunne = np.zeros([64,64,3])
    Imunne[:,:,0] = 1.0
    Imunne[:,:,1] = 0.4
    Imunne[:,:,2] = 0.0
    colors.append(Imunne)
    
    # NK
    Imunne = np.zeros([64,64,3])
    Imunne[:,:,0] = 0.2
    Imunne[:,:,1] = 0.4
    Imunne[:,:,2] = 1.0
    colors.append(Imunne)
    
    # B cell
    Imunne = np.zeros([64,64,3])
    Imunne[:,:,0] = 0.4
    Imunne[:,:,1] = 0.9
    Imunne[:,:,2] = 0.5
    colors.append(Imunne)
    
    # Neutrophils
    Imunne = np.zeros([64,64,3])
    Imunne[:,:,0] = 0.0
    Imunne[:,:,1] = 0.0
    Imunne[:,:,2] = 1.0
    colors.append(Imunne)
    
    # Macrophages
    Imunne = np.zeros([64,64,3])
    Imunne[:,:,0] = 0.3
    Imunne[:,:,1] = 0.8
    Imunne[:,:,2] = 1.0
    colors.append(Imunne)
    
    
    
    Imunne = np.zeros([64,64,3])
    Imunne[:,:,0] = 0.3
    Imunne[:,:,1] = 0.0
    Imunne[:,:,2] = 0.0
    colors.append(Imunne)
    
    
    Imunne = np.zeros([64,64,3])
    Imunne[:,:,0] = 0.5
    Imunne[:,:,1] = 0.1
    Imunne[:,:,2] = 0.0
    colors.append(Imunne)
    
    Imunne = np.zeros([64,64,3])
    Imunne[:,:,0] = 0.5
    Imunne[:,:,1] = 0.5
    Imunne[:,:,2] = 0.0
    colors.append(Imunne)
    
    
    # Other immune
    Imunne = np.zeros([64,64,3])
    Imunne[:,:,0] = 0.3
    Imunne[:,:,1] = 0.5
    Imunne[:,:,2] = 0.0
    colors.append(Imunne)
    
    cell_visual = np.zeros([64,64,3])
    
    for c in range(n_class):
        seg_color = seg[c:c+1].repeat(3, axis=0).transpose(1,2,0)
        cell_visual += seg_color * colors[c]
    
    return cell_visual.astype(float)