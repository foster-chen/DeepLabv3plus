from collections import namedtuple
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

def get_cs_trainId(mode):
    assert mode in [None, "carla"], "mode must be one of \"default\" and \"carla\""
    CityscapesClass = namedtuple('CityscapesClass', ['name', 'id', 'train_id', 'category', 'category_id',
                                                     'has_instances', 'ignore_in_eval', 'color'])
    
    if mode is None:
        classes = [
            CityscapesClass('unlabeled',            0,  255, 'void',         0, False, True,  (0, 0, 0)),
            CityscapesClass('ego vehicle',          1,  255, 'void',         0, False, True,  (0, 0, 0)),
            CityscapesClass('rectification border', 2,  255, 'void',         0, False, True,  (0, 0, 0)),
            CityscapesClass('out of roi',           3,  255, 'void',         0, False, True,  (0, 0, 0)),
            CityscapesClass('static',               4,  255, 'void',         0, False, True,  (0, 0, 0)),
            CityscapesClass('dynamic',              5,  255, 'void',         0, False, True,  (111, 74, 0)),
            CityscapesClass('ground',               6,  255, 'void',         0, False, True,  (81, 0, 81)),
            CityscapesClass('road',                 7,  0,   'flat',         1, False, False, (128, 64, 128)),
            CityscapesClass('sidewalk',             8,  1,   'flat',         1, False, False, (244, 35, 232)),
            CityscapesClass('parking',              9,  255, 'flat',         1, False, True,  (250, 170, 160)),
            CityscapesClass('rail track',           10, 255, 'flat',         1, False, True,  (230, 150, 140)),
            CityscapesClass('building',             11, 2,   'construction', 2, False, False, (70, 70, 70)),
            CityscapesClass('wall',                 12, 3,   'construction', 2, False, False, (102, 102, 156)),
            CityscapesClass('fence',                13, 4,   'construction', 2, False, False, (190, 153, 153)),
            CityscapesClass('guard rail',           14, 255, 'construction', 2, False, True,  (180, 165, 180)),
            CityscapesClass('bridge',               15, 255, 'construction', 2, False, True,  (150, 100, 100)),
            CityscapesClass('tunnel',               16, 255, 'construction', 2, False, True,  (150, 120, 90)),
            CityscapesClass('pole',                 17, 5,   'object',       3, False, False, (153, 153, 153)),
            CityscapesClass('polegroup',            18, 255, 'object',       3, False, True,  (153, 153, 153)),
            CityscapesClass('traffic light',        19, 6,   'object',       3, False, False, (250, 170, 30)),
            CityscapesClass('traffic sign',         20, 7,   'object',       3, False, False, (220, 220, 0)),
            CityscapesClass('vegetation',           21, 8,   'nature',       4, False, False, (107, 142, 35)),
            CityscapesClass('terrain',              22, 9,   'nature',       4, False, False, (152, 251, 152)),
            CityscapesClass('sky',                  23, 10,  'sky',          5, False, False, (70, 130, 180)),
            CityscapesClass('person',               24, 11,  'human',        6, True,  False, (220, 20, 60)),
            CityscapesClass('rider',                25, 12,  'human',        6, True,  False, (255, 0, 0)),
            CityscapesClass('car',                  26, 13,  'vehicle',      7, True,  False, (0, 0, 142)),
            CityscapesClass('truck',                27, 14,  'vehicle',      7, True,  False, (0, 0, 70)),
            CityscapesClass('bus',                  28, 15,  'vehicle',      7, True,  False, (0, 60, 100)),
            CityscapesClass('caravan',              29, 255, 'vehicle',      7, True,  True,  (0, 0, 90)),
            CityscapesClass('trailer',              30, 255, 'vehicle',      7, True,  True,  (0, 0, 110)),
            CityscapesClass('train',                31, 16,  'vehicle',      7, True,  False, (0, 80, 100)),
            CityscapesClass('motorcycle',           32, 17,  'vehicle',      7, True,  False, (0, 0, 230)),
            CityscapesClass('bicycle',              33, 18,  'vehicle',      7, True,  False, (119, 11, 32)),
            CityscapesClass('license plate',        -1, 255, 'vehicle',      7, False, True,  (0, 0, 142)),
        ]
        
    elif mode == "carla":
        classes = [
            CityscapesClass('unlabeled',            0,  255, 'void',         0, False, True,  (0, 0, 0)),
            CityscapesClass('ego vehicle',          1,  255, 'void',         0, False, True,  (0, 0, 0)),
            CityscapesClass('rectification border', 2,  255, 'void',         0, False, True,  (0, 0, 0)),
            CityscapesClass('out of roi',           3,  255, 'void',         0, False, True,  (0, 0, 0)),
            CityscapesClass('static',               4,  255, 'void',         0, False, True,  (0, 0, 0)),
            CityscapesClass('dynamic',              5,  255, 'void',         0, False, True,  (111, 74, 0)),
            CityscapesClass('ground',               6,  11,  'void',         0, False, True,  (81, 0, 81)),
            CityscapesClass('road',                 7,  4,   'flat',         1, False, False, (128, 64, 128)),
            CityscapesClass('sidewalk',             8,  5,   'flat',         1, False, False, (244, 35, 232)),
            CityscapesClass('parking',              9,  255, 'flat',         1, False, True,  (250, 170, 160)),
            CityscapesClass('rail track',           10, 13,  'flat',         1, False, True,  (230, 150, 140)),
            CityscapesClass('building',             11, 0,   'construction', 2, False, False, (70, 70, 70)),
            CityscapesClass('wall',                 12, 8,   'construction', 2, False, False, (102, 102, 156)),
            CityscapesClass('fence',                13, 1,   'construction', 2, False, False, (190, 153, 153)),
            CityscapesClass('guard rail',           14, 14,  'construction', 2, False, True,  (180, 165, 180)),
            CityscapesClass('bridge',               15, 12,  'construction', 2, False, True,  (150, 100, 100)),
            CityscapesClass('tunnel',               16, 255, 'construction', 2, False, True,  (150, 120, 90)),
            CityscapesClass('pole',                 17, 3,   'object',       3, False, False, (153, 153, 153)),
            CityscapesClass('polegroup',            18, 255, 'object',       3, False, True,  (153, 153, 153)),
            CityscapesClass('traffic light',        19, 15,  'object',       3, False, False, (250, 170, 30)),
            CityscapesClass('traffic sign',         20, 9,   'object',       3, False, False, (220, 220, 0)),
            CityscapesClass('vegetation',           21, 6,   'nature',       4, False, False, (107, 142, 35)),
            CityscapesClass('terrain',              22, 16,  'nature',       4, False, False, (152, 251, 152)),
            CityscapesClass('sky',                  23, 10,  'sky',          5, False, False, (70, 130, 180)),
            CityscapesClass('person',               24, 2,   'human',        6, True,  False, (220, 20, 60)),
            CityscapesClass('rider',                25, 2,   'human',        6, True,  False, (255, 0, 0)),
            CityscapesClass('car',                  26, 7,   'vehicle',      7, True,  False, (0, 0, 142)),
            CityscapesClass('truck',                27, 7,   'vehicle',      7, True,  False, (0, 0, 70)),
            CityscapesClass('bus',                  28, 7,   'vehicle',      7, True,  False, (0, 60, 100)),
            CityscapesClass('caravan',              29, 7,   'vehicle',      7, True,  True,  (0, 0, 90)),
            CityscapesClass('trailer',              30, 7,   'vehicle',      7, True,  True,  (0, 0, 110)),
            CityscapesClass('train',                31, 7,   'vehicle',      7, True,  False, (0, 80, 100)),
            CityscapesClass('motorcycle',           32, 7,   'vehicle',      7, True,  False, (0, 0, 230)),
            CityscapesClass('bicycle',              33, 7,   'vehicle',      7, True,  False, (119, 11, 32)),
            CityscapesClass('license plate',        -1, 255, 'vehicle',      7, False, True,  (0, 0, 142)),
        ]
    
    return classes


def get_carla_trainId():
    
    CarlaClass = namedtuple('CarlaClass', ['carla_name', 'carla_id', 'carla_color', 'train_id', 'cs_id', 'cs_color'])
    classes = [
        CarlaClass('unlabeled',       0,  (0, 0, 0),       255, 0,  (0, 0, 0)),
        CarlaClass('building',        1,  (70, 70, 70),    2,   11, (70, 70, 70)),
        CarlaClass('fence',           2,  (100, 40, 40),   4,   13, (190, 153, 153)),
        CarlaClass('other',           3,  (55, 90, 80),    255, 0,  (0, 0, 0)),
        CarlaClass('person',          4,  (220, 20, 60),   11,  24, (220, 20, 60)),
        CarlaClass('rider',           5,  (255, 0, 0),     12,  25, (255, 0, 0)),
        CarlaClass('pole',            6,  (153, 153, 153), 5,   17, (153, 153, 153)),
        CarlaClass('road line',       7,  (157, 234, 50),  0,   7,  (128, 64, 128)),
        CarlaClass('road',            8,  (128, 64, 128),  0,   7,  (128, 64, 128)),
        CarlaClass('sidewalk',        9,  (244, 35, 232),  1,   8,  (244, 35, 232)),
        CarlaClass('vegetation',      10, (107, 142, 35),  8,   21, (107, 142, 35)),
        CarlaClass('car',             11, (0, 0, 142),     13,  26, (0, 0, 142)),
        CarlaClass('truck',           12, (0, 0, 70),      14,  27, (0, 0, 70)),
        CarlaClass('bus',             13, (0, 60, 100),    15,  28, (0, 60, 100)),
        CarlaClass('train',           14, (0, 80, 100),    16,  31, (0, 80, 100)),
        CarlaClass('motorcycle',      15, (0, 0, 230),     17,  32, (0, 0, 230)),
        CarlaClass('bicycle',         16, (119, 11, 32),   18,  33, (119, 11, 32)),
        CarlaClass('wall',            17, (102, 102, 156), 3,   12, (102, 102, 156)),
        CarlaClass('traffic sign',    18, (220, 220, 0),   7,   20, (220, 220, 0)),
        CarlaClass('sky',             19, (70, 130, 180),  10,  23, (70, 130, 180)),
        CarlaClass('ground',          20, (81, 0, 81),     255, 6,  (81, 0, 81)),
        CarlaClass('bridge',          21, (150, 100, 100), 255, 15, (150, 100, 100)),
        CarlaClass('rail track',      22, (230, 150, 140), 255, 10, (230, 150, 140)),
        CarlaClass('guard rail',      23, (180, 165, 180), 255, 14, (180, 165, 180)),
        CarlaClass('traffic light',   24, (250, 170, 30),  6,   19, (250, 170, 30)),
        CarlaClass('static',          25, (110, 190, 160), 255, 4,  (0, 0, 0)),
        CarlaClass('dynamic',         26, (170, 120, 50),  255, 5,  (0, 0, 0)),
        CarlaClass('water',           27, (45, 60, 150),   255, 0,  (0, 0, 0)),
        CarlaClass('terrain',         28, (145, 170, 100), 9,   22, (152, 251, 152)),
        CarlaClass('general anomaly', 29, (236, 236, 236), 255, 0,  (0, 0, 0)),
    ]

    return classes

if __name__ == "__main__":

    def encode(input):   
        classes = get_carla_trainId()
        encoder = np.array([c.train_id for c in classes])
        return encoder[input]
    
    def decode(input):
        classes = get_carla_trainId()
        
        train_id_list = [c.train_id for c in classes]
        train_id_to_id = np.zeros(256, dtype="uint8")  # this is so that the value at index == 255 is id 0 which is unlabeled
        for i, j in enumerate(train_id_list):
            train_id_to_id[j] = i
        train_id_to_id[255] = 0
        
        return train_id_to_id[input]
    
    def to_color(input):
        classes = get_carla_trainId()
        id_to_color = [c.cs_color for c in classes]
        id_to_color = np.array(id_to_color)
        return id_to_color[input]

    im = np.array(Image.open("/home/chenht/datasets/Carla2Cityscapes/train/semantic/125-cloud0-pre0-fog0-sun70-n100-v100019_semantic.png"), dtype="uint8")
    # print(np.unique(im))
    # print(np.unique(encode(im)))
    # im = im[250:750, 500:1500]
    # print(im)
    # print(encode(im))
    # print(decode(encode(im)))
    
    im = Image.fromarray(to_color(decode(encode(im))).astype("uint8"))
    im.save("/home/chenht/DeepLabv3plus/DeepLabV3Plus-Pytorch/datasets/test.png")
    # plt.imshow(to_color(decode(encode(im))))
    # plt.axis("off")
    # plt.savefig("/home/chenht/DeepLabv3plus/DeepLabV3Plus-Pytorch/datasets/test.png")