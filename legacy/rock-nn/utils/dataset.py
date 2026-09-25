import torch
from torch.utils.data import DataLoader
from custom_dataset import CustomDataset


def load_data_from_file(file_name, pore_label="Poro"):
    content = []
    with open(file_name, "r") as f:
        for line in f.readlines():
            mod_line = line.strip("\n").split("\t")
            rgb = [int(i) for i in mod_line[0:3]]
            label = 1 if mod_line[3] == pore_label else 0
            rgb.append(label)
            content.append(rgb)
    return content

def split_dataset(dataset, ratio=0.8):
    train_size = int(ratio * len(dataset))
    test_size = len(dataset) - train_size
    return torch.utils.data.random_split(dataset, [train_size, test_size])

def create_dataloaders(data):
    dataset = CustomDataset(load_data_from_file(data))
    train_dataset, test_dataset = split_dataset(dataset)

    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    return train_dataloader, test_dataloader