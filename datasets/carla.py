import json
import os
from collections import namedtuple

import torch
import torch.utils.data as data
from PIL import Image
import numpy as np
import pandas as pd
from .label import get_carla_trainId


class Carla(data.Dataset):
    
    classes = get_carla_trainId()

    id_to_train_id = np.array([c.train_id for c in classes])

    train_id_list = [c.train_id for c in classes]
    train_id_to_id = np.zeros(256, dtype="uint8")  # this is so that the value at index == 255 is id 0 which is unlabeled
    
    for i, j in enumerate(train_id_list):
        train_id_to_id[j] = i
    train_id_to_id[255] = 0
        
    
    id_to_color = [c.cs_color for c in classes]
    id_to_color = np.array(id_to_color)

    def __init__(self, root, split='train', transform=None):
        self.root = os.path.expanduser(root)
        self.split = split
        self.tranform = transform
        self.df = self.create_df(self.root, self.split)
        # print(self.df.iloc[:5, 0])
        # print(self.df.iloc[:5, 1])
        # print(self.df.iloc[:5, 2])
        # data_dir = "/home/chenht/datasets/NightLab/"
    
    @classmethod
    def get_file_paths(cls, directory):  # helper function to get absolute paths for all files within a directory
        file_paths = []
        for dirpath, folders, filenames in os.walk(directory):
            for f in filenames:
                file_paths.append(os.path.abspath(os.path.join(dirpath, f)))
        return file_paths

    @classmethod
    def create_df(cls, data_dir, split):
        train_file_names = sorted([filename[:-4] for filename in os.listdir(data_dir + split + "/rgb")])
        train_data_list = sorted(cls.get_file_paths(data_dir + split + "/rgb"))
        train_label_list = sorted(cls.get_file_paths(data_dir + split + "/semantic"))

        return pd.DataFrame(dict(file_name=train_file_names, data_path=train_data_list, label_path=train_label_list))


    def __getitem__(self, index):
        image = Image.open(self.df.data_path[index])
        target = Image.open(self.df.label_path[index])
        if self.tranform:
            image, target = self.tranform(image, target)
        target = self.encode_target(target)
        return image, target
    
    def __len__(self):
        return len(self.df)
    
    @classmethod
    def encode_target(cls, target):
        return cls.id_to_train_id[np.array(target)]

    @classmethod
    def decode_target(cls, target):
        #target = target.astype('uint8') + 1
        target_to_id = cls.train_id_to_id[target]
        id_to_color = cls.id_to_color[target_to_id]
        return id_to_color
# pd.set_option('display.max_colwidth', None) 
# a = Carla("/home/chenht/datasets/Carla2Cityscapes/")

