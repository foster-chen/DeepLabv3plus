import json
import os
from collections import namedtuple

import torch
import torch.utils.data as data
from PIL import Image
import numpy as np
import pandas as pd
from .label import get_carla_trainId, get_cs_trainId
from tqdm import tqdm


class ACDC(data.Dataset):

    def __init__(self, root, scene, split='train', coder=None, transform=None):
        
        self.classes = get_cs_trainId(None)
        train_id_to_color = [c.color for c in self.classes if (c.train_id != -1 and c.train_id != 255)]
        train_id_to_color.append([0, 0, 0])
        self.train_id_to_color = np.array(train_id_to_color)
        self.id_to_train_id = np.array([c.train_id for c in self.classes])
        
        self.root = os.path.expanduser(root)
        assert scene in ['rain', 'snow', 'fog', 'night']
        self.scene = scene
        self.split = split
        self.tranform = transform
        self.df = self.create_df(self.root, self.scene, self.split)
        self.class_weights = self._get_class_weights()
        # print(self.df.iloc[:5, 0])
        # print(self.df.iloc[:5, 1])
        # print(self.df.iloc[:5, 2])
        # data_dir = "/home/chenht/datasets/NightLab/"
    
    def _get_class_weights(self):
        try:
            class_weights = np.load(os.path.join(self.root, "gt", self.scene, 'class_weights.npy'))
            return class_weights
        except FileNotFoundError:
            print("class_weights.npy not found, computing class weights...")
            label_counts = np.zeros(shape=(20), dtype='int64')
            for filename in tqdm(self.df.iloc[:, 1], desc="Caculating"):
                im = np.array(Image.open(filename))
                im[im == 255] = 19
                classes, counts = np.unique(im, return_counts=True)
                label_counts[classes] += counts
            train_id_counts = label_counts + 1
            class_weights = np.sum(train_id_counts) / (19 * train_id_counts)
            np.save(os.path.join(self.root, "gt", self.scene, 'class_weights.npy'), class_weights[:-1])
            return class_weights
    
    @classmethod
    def get_file_paths(cls, directory):  # helper function to get absolute paths for all files within a directory
        file_paths = []
        for dirpath, folders, filenames in os.walk(directory):
            for f in filenames:
                file_paths.append(os.path.abspath(os.path.join(dirpath, f)))
        return file_paths

    @classmethod
    def create_df(cls, data_dir, scene, split):
        data_list = sorted(cls.get_file_paths(os.path.join(data_dir, "rgb_anon", scene, split)))
        label_list = sorted([filename for filename in cls.get_file_paths(os.path.join(data_dir, "gt", scene, split)) if "TrainIds" in filename])

        return pd.DataFrame(dict(data_path=data_list, label_path=label_list))


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

if __name__ == "__main__":
    pd.set_option('display.max_colwidth', None)
    dataset = ACDC("/home/gaoha/epe/ACDC", "snow", 'val')
    print(dataset.df.head())