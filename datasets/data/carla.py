import json
import os
from collections import namedtuple

import torch
import torch.utils.data as data
from PIL import Image
import numpy as np
import pandas as pd
from label import get_carla_trainId


class Carla(data.Dataset):

    #train_id_to_color = [(0, 0, 0), (128, 64, 128), (70, 70, 70), (153, 153, 153), (107, 142, 35),
    #                      (70, 130, 180), (220, 20, 60), (0, 0, 142)]
    #train_id_to_color = np.array(train_id_to_color)
    #id_to_train_id = np.array([c.category_id for c in classes], dtype='uint8') - 1

    classes = get_carla_trainId()

    train_id_to_color = [c.color for c in classes if (c.train_id != -1 and c.train_id != 255)]
    train_id_to_color.append([0, 0, 0])
    train_id_to_color = np.array(train_id_to_color)
    id_to_train_id = np.array([c.train_id for c in classes])
    

    def __init__(self, root, split='train', mode='fine', target_type='semantic', transform=None):
        self.root = os.path.expanduser(root)
        self.split = split
        self.tranform = transform
        self.df = self.create_df(self.root, self.split)
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
        train_file_names = sorted([filename[:-4] for filename in os.listdir(data_dir + split + "/image")])
        train_data_list = sorted(cls.get_file_paths(data_dir + split + "/image"))
        train_label_list = sorted(cls.get_file_paths(data_dir + split + "/label_trainId"))

        return pd.DataFrame(dict(file_name=train_file_names, data_path=train_data_list, label_path=train_label_list))


    def __getitem__(self, index):
        image = Image.open(self.df.data_path[index])
        target = np.array(Image.open(self.df.label_path[index]).resize((1024, 512), resample=Image.NEAREST))
        target[target == -1] = 255
        target = Image.fromarray(target)
        # print(f"#################### image size: {image.size}, target size: {target.size}")
        if self.tranform:
            image, target = self.tranform(image, target)
        
        return image, target
    
    def __len__(self):
        return len(self.df)
    
    @classmethod
    def encode_target(cls, target):
        return cls.id_to_train_id[np.array(target)]

    @classmethod
    def decode_target(cls, target):
        target[target == 255] = 19
        #target = target.astype('uint8') + 1
        return cls.train_id_to_color[target]