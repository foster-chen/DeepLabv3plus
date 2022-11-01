import enum
from PIL import Image
import numpy as np
import os
import shutil
import argparse
from collections import namedtuple
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def get_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--srs_dir", type=str,default=None,
                        help="Source delivery directory")
    parser.add_argument("--dst_dir", type=str, default="/DATA_EDS/chenht/datasets",
                        help="destination directory")
    parser.add_argument("--dataset_name", type=str, default=None,
                        help="new folder to store pre-processed images")
    parser.add_argument("--label_only", action="store_true", default=False,
                        help="only process the label maps and not copy rgb images")
    return parser


def carla_color2label(color_map):
    height, width, _ = color_map.shape
    label_map = np.zeros(shape=(height, width, 35))
    color2id = get_coder()
    for idx, (color, id) in enumerate(color2id.items()):
        label_map[:, :, color2id[tuple(color)] ] += (color_map == color).all(axis=2) * id
    label_map = label_map.sum(axis=2).astype('uint8')
    return label_map   


def get_coder():
    CityscapesClass = namedtuple('CityscapesClass', ['name', 'id', 'train_id', 'category', 'category_id',
                                                     'has_instances', 'ignore_in_eval', 'color'])
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
    
    name2id = {c.name: c.id for c in classes}
    name2color = {
                'unlabeled': (0, 0, 0), ###
                'building': (70, 70, 70), ###
                'fence': (100, 40, 40), ###
                'other': (55, 90, 80),
                'person': (220, 20, 60), ###
                'pole': (153, 153, 153), ###
                'road line': (157, 234, 50),
                'road': (128, 64, 128), ###
                'sidewalk': (244, 35, 232), ###
                'vegetation': (107, 142, 35), ###
                'car': (0, 0, 142), ###
                'wall': (102, 102, 156), ###
                'traffic sign': (220, 220, 0), ###
                'sky': (70, 130, 180), ###
                'ground': (81, 0, 81), ###
                'bridge': (150, 100, 100), ###
                'rail track': (230, 150, 140), ###
                'guard rail': (180, 165, 180), ###
                'traffic light': (250, 170, 30), ###
                'static': (110, 190, 160), ###
                'dynamic': (170, 120, 50),
                'water': (45, 60, 150), ###
                'terrain': (145, 170, 100), ###
                'anomaly': (255,255,255), ###
                'rider': (255, 0, 0), ###
                'truck': (0, 0, 70), ###
                'bus': (0, 60, 100), ###
                'train': (0, 80, 100), ###
                'motorcycle': (0, 0, 230), ###
                'bicycle': (119, 11, 32), ###
            }
    color2name = {j: i for i, j in name2color.items()}
    for color in color2name:
        if color2name[color] == 'road line':
            color2name[color] = 'road'
        elif color2name[color] == 'anomaly':
            color2name[color] = 'dynamic'
        elif color2name[color] == 'other':
            color2name[color] = 'unlabeled'
        elif color2name[color] == 'water':
            color2name[color] = 'unlabeled'
    color2id = {color: name2id[color2name[color]] for color in color2name}
    return color2id


def prep_dst_folder(dst_path):
    os.makedirs(os.path.join(dst_path, "train", "semantic"), exist_ok=True)
    os.makedirs(os.path.join(dst_path, "train", "rgb"), exist_ok=True)
    os.makedirs(os.path.join(dst_path, "val", "semantic"), exist_ok=True)
    os.makedirs(os.path.join(dst_path, "val", "rgb"), exist_ok=True)
    

def main():
    opts = get_argparser().parse_args()
    assert opts.srs_dir is not None and opts.dataset_name is not None, "use --srs_dir and --dataset_name to specify delivery and destination folders"
    
    opts.dst_dir = os.path.join(opts.dst_dir, opts.dataset_name)
    srs_filenames = [name[:-8] for name in os.listdir(opts.srs_dir) if "semantic" not in name]
    filename_train, filename_val = train_test_split(srs_filenames, test_size=0.3, random_state=42)
    prep_dst_folder(opts.dst_dir)
    for file in tqdm(filename_train, desc="Processing train set"):
        color_map = np.array(Image.open(os.path.join(opts.srs_dir, f"{file}_semantic.png")))
        label_map = carla_color2label(color_map)
        Image.fromarray(label_map).save(os.path.join(opts.dst_dir, "train", "semantic", f"{file}_semantic.png"))
        if not opts.label_only:
            shutil.copyfile(os.path.join(opts.srs_dir, f"{file}_rgb.png"), os.path.join(opts.dst_dir, "train", "rgb", f"{file}_rgb.png"))
            
    for file in tqdm(filename_val, desc="Processing val set"):
        color_map = np.array(Image.open(os.path.join(opts.srs_dir, f"{file}_semantic.png")))
        label_map = carla_color2label(color_map)
        Image.fromarray(label_map).save(os.path.join(opts.dst_dir, "val", "semantic", f"{file}_semantic.png"))
        if not opts.label_only:
            shutil.copyfile(os.path.join(opts.srs_dir, f"{file}_rgb.png"), os.path.join(opts.dst_dir, "val", "rgb", f"{file}_rgb.png"))
    

if __name__ == "__main__":
    main()
