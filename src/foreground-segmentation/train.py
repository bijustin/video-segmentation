#!/usr/bin/env python3

import glob
import os
from torch.utils.data import Dataset, DataLoader
import torch
from skimage import io
from Netmodel import SegNet

MAX_EPOCHS = 100
MODEL_PATH = 'training_model/'
OUTPUT_CHANNEL = 3

class Trainingset(Dataset):

    def __init__(self, root_path = 'dataset/dataset2014/dataset'):
        """
        Get paths of images and their ground truths 
        """
        if not os.path.isdir(root_path):
            # download and unzip
            pass
        self.img_path = sorted(glob.glob(os.path.join(root_path, '*', '*', 'input', '*jpg')))
        self.gt_path = sorted(glob.glob(os.path.join(root_path, '*', '*', 'groundtruth', '*.png')))

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        curr_img_path = self.img_path[idx]
        curr_gt_path = self.gt_path[idx]
        img = io.imread(curr_img_path)
        gt = io.imread(curr_gt_path)
        sample = {'image': img, 'gt': gt}

        return sample

    def __len__(self):
        """
        length of the dataset 
        """
        return len(self.img_path)


def getData():
    # CDnet2014 dataset for now
    training_set = Trainingset()
    loader = DataLoader(training_set, batch_size=16, shuffle=True, num_workers=4)

    return loader

def train(loader, clear = True):

    model = None

    if not clear:
        try:
            model = torch.load(MODEL_PATH)
        except:
            model = SegNet(OUTPUT_CHANNEL)
    else:
        model = SegNet(OUTPUT_CHANNEL)
    

    for epoch in range(MAX_EPOCHS):
        for img, gt in loader:
            # train the network
            pass

    model.save(model.state_dict(), MODEL_PATH)


if __name__ == "__main__":
    loader = getData()
    train(loader)


