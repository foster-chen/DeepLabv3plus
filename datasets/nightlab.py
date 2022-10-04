import json
import os
from collections import namedtuple

import torch
import torch.utils.data as data
from PIL import Image
import numpy as np
import pandas as pd
from .label import get_cs_trainId


class NightLab(data.Dataset):

    def __init__(self, root, split='train', coder=None, transform=None):
        self.classes = get_cs_trainId(coder)
        train_id_to_color = [c.color for c in self.classes if (c.train_id != -1 and c.train_id != 255)]
        train_id_to_color.append([0, 0, 0])
        self.train_id_to_color = np.array(train_id_to_color)
        self.id_to_train_id = np.array([c.train_id for c in self.classes])
        
        self.root = os.path.expanduser(root)
        self.split = split
        self.tranform = transform
        self.df = self.create_df(self.root, self.split)
        self.class_weights = _get_class_weights()
        # data_dir = "/home/chenht/datasets/NightLab/"
    
    def _get_class_weights(self):
        try:
            class_weights = np.load(os.path.join(self.root, 'class_weights.npy'))
            return class_weights
        except FileNotFoundError:
            print("class_weights.npy not found, computing class weights...")
            label_counts = np.zeros(shape=(34), dtype='int64')
            for filename in tqdm(self.df.iloc[:, 2], desc="Caculating"):
                im = np.array(Image.open(filename))
                im[im == -1] = 0
                classes, counts = np.unique(im, return_counts=True)
                label_counts[classes] += counts
            train_id_counts = label_counts[c.id for c in classes if c.train_id != 255] + 1
            class_weights = np.sum(train_id_counts) / (19 * train_id_counts)
            np.save(os.path.join(self.root, "class_weights.npy"), class_weights)
            return class_weights
    
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
        train_label_list = sorted(cls.get_file_paths(data_dir + split + "/label"))

        return pd.DataFrame(dict(file_name=train_file_names, data_path=train_data_list, label_path=train_label_list))


    def __getitem__(self, index):
        image = Image.open(self.df.data_path[index])
        target = Image.open(self.df.label_path[index]).resize((1024, 512), resample=Image.NEAREST)
        # print(f"#################### image size: {image.size}, target size: {target.size}")
        if self.tranform:
            image, target = self.tranform(image, target)
        target = self.encode_target(target)
        return image, target
    
    def __len__(self):
        return len(self.df)
    
    def encode_target(self, target):
        return self.id_to_train_id[np.array(target)]

    def decode_target(self, target):
        target[target == 255] = 19
        #target = target.astype('uint8') + 1
        return self.train_id_to_color[target]