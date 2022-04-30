import torch
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, content_data):
        self.content_data = content_data
    
    def __len__(self):
        return len(self.content_data)
    
    def __getitem__(self, index):
        data = self.content_data[index]
        rgb = torch.Tensor(data[0:3])
        label = data[3]
        
        return rgb, label