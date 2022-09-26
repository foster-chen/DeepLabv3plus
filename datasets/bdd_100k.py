import json
import os
from collections import namedtuple

import torch
import torch.utils.data as data
from PIL import Image
import numpy as np
import pandas as pd
from .label import get_carla_trainId, get_cs_trainId


class BDD_100K(data.Dataset):

    def __init__(self, root, split='train', coder=None, transform=None):
        
        self.classes = get_cs_trainId(None)
        train_id_to_color = [c.color for c in self.classes if (c.train_id != -1 and c.train_id != 255)]
        train_id_to_color.append([0, 0, 0])
        self.train_id_to_color = np.array(train_id_to_color)
        self.id_to_train_id = np.array([c.train_id for c in self.classes])
        
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
        file_names = sorted([filename[:-4] for filename in os.listdir(os.path.join(data_dir, "images", "10k", split))])
        data_list = sorted(cls.get_file_paths(os.path.join(data_dir, "images", "10k", split)))
        label_list = sorted(cls.get_file_paths(os.path.join(data_dir, "labels", "sem_seg", "masks", split)))

        return pd.DataFrame(dict(file_name=file_names, data_path=data_list, label_path=label_list))


    def __getitem__(self, index):
        image = Image.open(self.df.data_path[index])
        target = Image.open(self.df.label_path[index])
        if self.tranform:
            image, target = self.tranform(image, target)
        # target = self.encode_target(target)
        return image, target
    
    def __len__(self):
        return len(self.df)
    
    def encode_target(self, target):
        return self.id_to_train_id[np.array(target)]

    def decode_target(self, target):
        target[target == 255] = 19
        return self.train_id_to_color[target]
    
        #target = target.astype('uint8') + 1
        # target_to_id = cls.train_id_to_id[target]
        # id_to_color = cls.id_to_color[target_to_id]
        # return id_to_color
# pd.set_option('display.max_colwidth', None) 
# a = Carla("/home/chenht/datasets/Carla2Cityscapes/")

