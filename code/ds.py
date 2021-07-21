import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms



class MIBIDataset(Dataset):

    def __init__(self, data, transform=None):
        """
        Parameters: data is of shape(n_cells, img_h, img_w, n_channels)

        """
        self.data = data
        self.transform = transforms.Compose([transforms.ToTensor()])


    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        seg = self.data[idx][0]
        real = self.data[idx][1]
        
        seg = self.transform(seg)
        real = self.transform(real)
        
        return seg, real