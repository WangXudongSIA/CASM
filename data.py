import os
import torch
import numpy as np
from PIL import Image
import torch.utils.data as data

def populate_train_list(orig_images_path, hazy_images_path):
    image_list_haze = ['%s%s' % (hazy_images_path, img_name) for img_name in os.listdir(hazy_images_path)]
    image_list_orig = ['%s%s' % (orig_images_path, img_name) for img_name in os.listdir(orig_images_path)]
    return image_list_haze, image_list_orig

def populate_validation_list(orig_images_path, hazy_images_path):
    image_list_haze = ['%s%s' % (hazy_images_path, img_name) for img_name in os.listdir(hazy_images_path)]
    image_list_orig = ['%s%s' % (orig_images_path, img_name) for img_name in os.listdir(orig_images_path)]
    return image_list_haze, image_list_orig

class MyDataset(data.Dataset):
    def __init__(self, orig_images_path, hazy_images_path):
        self.xtrain_list, self.ytrain_list = populate_train_list(orig_images_path, hazy_images_path)
        print('Total training examples:', len(self.xtrain_list))

    def __getitem__(self, index):
        data_orig_path = self.ytrain_list[index]
        data_hazy_path = self.xtrain_list[index]
        data_orig = Image.open(data_orig_path).resize((1920, 1080), Image.ANTIALIAS)  # 480 640
        data_hazy = Image.open(data_hazy_path).resize((1920, 1080), Image.ANTIALIAS)

        data_orig = np.array(data_orig) / 255.0
        data_hazy = np.array(data_hazy) / 255.0

        data_orig = torch.from_numpy(data_orig).float()
        data_hazy = torch.from_numpy(data_hazy).float()

        return data_orig.permute(2, 0, 1), data_hazy.permute(2, 0, 1)

    def __len__(self):
        return len(self.xtrain_list)

class Myval_Dataset(data.Dataset):
    def __init__(self, val_orig_images_path, val_hazy_images_path):
        self.xval_list, self.yval_list = populate_validation_list(val_orig_images_path, val_hazy_images_path)
        print('Total evaluating examples:', len(self.xval_list))

    def __getitem__(self, index):
        data_orig_path = self.yval_list[index]
        data_hazy_path = self.xval_list[index]
        data_orig = Image.open(data_orig_path).resize((640, 480), Image.ANTIALIAS)  # 480 640
        data_hazy = Image.open(data_hazy_path).resize((640, 480), Image.ANTIALIAS)

        data_orig = np.array(data_orig) / 255.0
        data_hazy = np.array(data_hazy) / 255.0

        data_orig = torch.from_numpy(data_orig).float()
        data_hazy = torch.from_numpy(data_hazy).float()

        return data_orig.permute(2, 0, 1), data_hazy.permute(2, 0, 1)

    def __len__(self):
        return len(self.xval_list)


