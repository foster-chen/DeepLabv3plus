import os
from webbrowser import get
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from collections import namedtuple


def get_carla_trainId():
    
    CarlaClass = namedtuple('CarlaClass', ['carla_name', 'carla_id', 'carla_color', 'train_id', 'cs_id', 'cs_color'])
    classes = [
        CarlaClass('unlabeled',       0,  (0, 0, 0),       255, 0,  (0, 0, 0)),
        CarlaClass('building',        1,  (70, 70, 70),    0,   11, (70, 70, 70)),
        CarlaClass('fence',           2,  (100, 40, 40),   1,   13, (190, 153, 153)),
        CarlaClass('other',           3,  (55, 90, 80),    255, 0,  (0, 0, 0)),
        CarlaClass('pedestrian',      4,  (220, 20, 60),   2,   24, (220, 20, 60)),
        CarlaClass('pole',            5,  (153, 153, 153), 3,   17, (153, 153, 153)),
        CarlaClass('road line',       6,  (157, 234, 50),  4,   7,  (128, 64, 128)),
        CarlaClass('road',            7,  (128, 64, 128),  4,   7,  (128, 64, 128)),
        CarlaClass('sidewalk',        8,  (244, 35, 232),  5,   8,  (244, 35, 232)),
        CarlaClass('vegetation',      9,  (107, 142, 35),  6,   21, (107, 142, 35)),
        CarlaClass('vehicle',         10, (0, 0, 142),     7,   26, (0, 0, 142)),
        CarlaClass('wall',            11, (102, 102, 156), 8,   12, (102, 102, 156)),
        CarlaClass('traffic sign',    12, (220, 220, 0),   9,   20, (220, 220, 0)),
        CarlaClass('sky',             13, (70, 130, 180),  10,  23, (70, 130, 180)),
        CarlaClass('ground',          14, (81, 0, 81),     11,  6,  (81, 0, 81)),
        CarlaClass('bridge',          15, (150, 100, 100), 12,  15, (150, 100, 100)),
        CarlaClass('rail track',      16, (230, 150, 140), 13,  0,  (0, 0, 0)),
        CarlaClass('guard rail',      17, (180, 165, 180), 14,  14, (180, 165, 180)),
        CarlaClass('traffic light',   18, (250, 170, 30),  15,  19, (250, 170, 30)),
        CarlaClass('static',          19, (110, 190, 160), 255, 4,  (0, 0, 0)),
        CarlaClass('dynamic',         20, (170, 120, 50),  255, 5,  (111, 74, 0)),
        CarlaClass('water',           21, (45, 60, 150),   255, 0,  (0, 0, 0)),
        CarlaClass('terrain',         22, (145, 170, 100), 16,  22, (152, 251, 152)),
        CarlaClass('general anomaly', 23, (236, 236, 236), 255, 0,  (0, 0, 0)),
    ]

    return classes


carla_dir = "/home/gaoha/epe/Delivery/Carla2Cityscapes_v1"

filenames = os.listdir(carla_dir)

labels = [os.path.join(carla_dir, filename) for filename in filenames if "semantic" in filename]
print(labels[0])

im = np.array(Image.open(labels[0]))


carla_map = get_carla_trainId()

