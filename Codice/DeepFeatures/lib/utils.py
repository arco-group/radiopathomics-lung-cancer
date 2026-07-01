import os
from csv import writer

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


def append_list_as_row(file_name, list_of_elem):
    # Open file in append mode
    with open(file_name, 'a+', newline='') as write_obj:
        # Create a writer object from csv module
        csv_writer = writer(write_obj, delimiter=';')
        # Add contents of list as last row in the csv file
        csv_writer.writerow(list_of_elem)

class PatchDataset(Dataset):
    """Patch dataset."""

    def __init__(self, csv_file, root_dir, patient_filter=None, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with labels.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        if patient_filter is not None:
            self.labels = pd.read_csv(csv_file, sep=";")
            self.labels = self.labels[self.labels["file_name"].str.contains(
                '|'.join(patient_filter))]
        else:
            self.labels = pd.read_csv(csv_file, sep=";")

        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        file_name = os.path.join(self.root_dir,
                                 self.labels.iloc[idx, 0].split("\\")[0],
                                 self.labels.iloc[idx, 0].split("\\")[1])

        img = cv2.imread(file_name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        label = self.labels.iloc[idx, 1]

        # sample = {'image': img, 'label': label}
        if self.transform:
            img = self.transform(img)

        return img, label


def split_dataloader(csv_file, root_dir, id_patients, train_index, test_index):
    X_train = PatchDataset(csv_file, root_dir, patient_filter=id_patients[train_index],
                           transform=transforms.Compose([transforms.ToTensor(),
                                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]))
    X_test = PatchDataset(csv_file, root_dir, patient_filter=id_patients[test_index],
                          transform=transforms.Compose([transforms.ToTensor(),
                                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]))

    train_loader = torch.utils.data.DataLoader(
        X_train, batch_size=64, shuffle=True, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(
        X_test, batch_size=64, shuffle=True, pin_memory=True)
    return train_loader, test_loader
