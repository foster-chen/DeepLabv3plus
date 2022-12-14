import json
import os
from collections import namedtuple
import random

import torch
import torch.utils.data as data
from PIL import Image
import numpy as np
from .label import get_cs_trainId
from tqdm import tqdm


class Cityscapes(data.Dataset):
    """Cityscapes <http://www.cityscapes-dataset.com/> Dataset.
    
    **Parameters:**
        - **root** (string): Root directory of dataset where directory 'leftImg8bit' and 'gtFine' or 'gtCoarse' are located.
        - **split** (string, optional): The image split to use, 'train', 'test' or 'val' if mode="gtFine" otherwise 'train', 'train_extra' or 'val'
        - **mode** (string, optional): The quality mode to use, 'gtFine' or 'gtCoarse' or 'color'. Can also be a list to output a tuple with all specified target types.
        - **transform** (callable, optional): A function/transform that takes in a PIL image and returns a transformed version. E.g, ``transforms.RandomCrop``
        - **target_transform** (callable, optional): A function/transform that takes in the target and transforms it.
    """

    # Based on https://github.com/mcordts/cityscapesScripts
    
    #train_id_to_color = [(0, 0, 0), (128, 64, 128), (70, 70, 70), (153, 153, 153), (107, 142, 35),
    #                      (70, 130, 180), (220, 20, 60), (0, 0, 142)]
    #train_id_to_color = np.array(train_id_to_color)
    #id_to_train_id = np.array([c.category_id for c in classes], dtype='uint8') - 1

    def __init__(self, root, split='train', mode='fine', target_type='semantic', coder=None, transform=None, subsample=None):
        self.classes = get_cs_trainId(coder)
        train_id_to_color = [c.color for c in self.classes if (c.train_id != -1 and c.train_id != 255)]
        train_id_to_color.append([0, 0, 0])
        self.train_id_to_color = np.array(train_id_to_color)
        self.id_to_train_id = np.array([c.train_id for c in self.classes])
        
        self.root = os.path.expanduser(root)
        self.mode = 'gtFine'
        self.target_type = target_type
        self.images_dir = os.path.join(self.root, 'leftImg8bit', split)

        self.targets_dir = os.path.join(self.root, self.mode, split)
        self.transform = transform

        self.split = split
        self.images = []
        self.targets = []
        self.subsample = subsample

        if split not in ['train', 'test', 'val']:
            raise ValueError('Invalid split for mode! Please use split="train", split="test"'
                             ' or split="val"')

        if not os.path.isdir(self.images_dir) or not os.path.isdir(self.targets_dir):
            raise RuntimeError('Dataset not found or incomplete. Please make sure all required folders for the'
                               ' specified "split" and "mode" are inside the "root" directory')
        
        for city in os.listdir(self.images_dir):
            img_dir = os.path.join(self.images_dir, city)
            target_dir = os.path.join(self.targets_dir, city)

            for file_name in os.listdir(img_dir):
                self.images.append(os.path.join(img_dir, file_name))
                target_name = '{}_{}'.format(file_name.split('_leftImg8bit')[0],
                                             self._get_target_suffix(self.mode, self.target_type))
                self.targets.append(os.path.join(target_dir, target_name))
                
        self.class_weights = self._get_class_weights()
                
        if self.subsample:
            random.seed(42)
            sampled = random.sample(list(np.arange(0, len(self.images), 1)), round(self.subsample * len(self.images)))
            self.images = list(np.array(self.images, dtype='object')[sampled])
            self.targets = list(np.array(self.targets, dtype='object')[sampled])
                
    def _get_class_weights(self):
        try:
            class_weights = np.load(os.path.join(self.root, 'class_weights.npy'))
            return class_weights
        except FileNotFoundError:
            print("class_weights.npy not found, computing class weights...")
            label_counts = np.zeros(shape=(34), dtype='int64')
            for filename in tqdm(self.targets, desc="Caculating", total=len(self.targets)):
                im = np.array(Image.open(filename))
                im[im == -1] = 0
                classes, counts = np.unique(im, return_counts=True)
                label_counts[classes] += counts
            train_id_counts = label_counts[[c.id for c in self.classes if c.train_id != 255]] + 1
            class_weights = np.sum(train_id_counts) / (19 * train_id_counts)
            np.save(os.path.join(self.root, "class_weights.npy"), class_weights)
            return label_counts

    def encode_target(self, target):
        return self.id_to_train_id[np.array(target)]

    def decode_target(self, target):
        target[target == 255] = 19
        #target = target.astype('uint8') + 1
        return self.train_id_to_color[target]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is a tuple of all target types if target_type is a list with more
            than one item. Otherwise target is a json object if target_type="polygon", else the image segmentation.
        """
        image = Image.open(self.images[index]).convert('RGB')
        target = Image.open(self.targets[index])
        # print(f"#################### image size: {np.array(image).shape}, target size: {np.array(target).shape}")
        if self.transform:
            image, target = self.transform(image, target)
            
        # if not isinstance(image, np.ndarray):
        #     image = np.array(image)
        target = self.encode_target(target)
        return image, target

    def __len__(self):
        return len(self.images)

    def _load_json(self, path):
        with open(path, 'r') as file:
            data = json.load(file)
        return data

    def _get_target_suffix(self, mode, target_type):
        if target_type == 'instance':
            return '{}_instanceIds.png'.format(mode)
        elif target_type == 'semantic':
            return '{}_labelIds.png'.format(mode)
        elif target_type == 'color':
            return '{}_color.png'.format(mode)
        elif target_type == 'polygon':
            return '{}_polygons.json'.format(mode)
        elif target_type == 'depth':
            return '{}_disparity.png'.format(mode)
        

if __name__ == "__main__":
    
    dataset = Cityscapes(root="/home/chenht/datasets/CityScapes",
                               split='val')
    loader = data.DataLoader(dataset, batch_size=8, shuffle=True, drop_last=True)
    iterator = iter(loader)
    while True:
        next(iterator)