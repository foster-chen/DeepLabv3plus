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
    parser.add_argument("--srs_dir", type=str, default=None, help="Source delivery directory")
    parser.add_argument("--dst_dir", type=str, default="/DATA_EDS/chenht/datasets", help="destination directory")
    parser.add_argument("--dataset_name", type=str, default=None, help="new folder to store pre-processed images")
    parser.add_argument("--label_only",
                        action="store_true",
                        default=False,
                        help="only process the label maps and not copy rgb images")
    parser.add_argument("--from_file", type=str, default=None, help="pre-compiled txt list of data directories.")
    return parser


def get_coder():
    Class = namedtuple(
        'CityscapesClass',
        ['name', 'id', 'train_id', 'category', 'category_id', 'has_instances', 'ignore_in_eval', 'color'])
    # yapf: disable
    classes = [
        #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
        Class(  'unlabeled'            ,  0 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
        Class(  'ego vehicle'          ,  1 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
        Class(  'rectification border' ,  2 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
        Class(  'out of roi'           ,  3 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
        Class(  'static'               ,  4 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
        Class(  'dynamic'              ,  5 ,      255 , 'void'            , 0       , False        , True         , (111, 74,  0) ),
        Class(  'ground'               ,  6 ,      255 , 'void'            , 0       , False        , True         , ( 81,  0, 81) ),
        Class(  'road'                 ,  7 ,        0 , 'flat'            , 1       , False        , False        , (128, 64,128) ),
        Class(  'sidewalk'             ,  8 ,        1 , 'flat'            , 1       , False        , False        , (244, 35,232) ),
        Class(  'parking'              ,  9 ,      255 , 'flat'            , 1       , False        , True         , (250,170,160) ),
        Class(  'rail track'           , 10 ,      255 , 'flat'            , 1       , False        , True         , (230,150,140) ),
        Class(  'building'             , 11 ,        2 , 'construction'    , 2       , False        , False        , ( 70, 70, 70) ),
        Class(  'wall'                 , 12 ,        3 , 'construction'    , 2       , False        , False        , (102,102,156) ),
        Class(  'fence'                , 13 ,        4 , 'construction'    , 2       , False        , False        , (190,153,153) ),
        Class(  'guard rail'           , 14 ,      255 , 'construction'    , 2       , False        , True         , (180,165,180) ),
        Class(  'bridge'               , 15 ,      255 , 'construction'    , 2       , False        , True         , (150,100,100) ),
        Class(  'tunnel'               , 16 ,      255 , 'construction'    , 2       , False        , True         , (150,120, 90) ),
        Class(  'pole'                 , 17 ,        5 , 'object'          , 3       , False        , False        , (153,153,153) ),
        Class(  'polegroup'            , 18 ,      255 , 'object'          , 3       , False        , True         , (153,153,153) ),
        Class(  'traffic light'        , 19 ,        6 , 'object'          , 3       , False        , False        , (250,170, 30) ),
        Class(  'traffic sign'         , 20 ,        7 , 'object'          , 3       , False        , False        , (220,220,  0) ),
        Class(  'vegetation'           , 21 ,        8 , 'nature'          , 4       , False        , False        , (107,142, 35) ),
        Class(  'terrain'              , 22 ,        9 , 'nature'          , 4       , False        , False        , (152,251,152) ),
        Class(  'sky'                  , 23 ,       10 , 'sky'             , 5       , False        , False        , ( 70,130,180) ),
        Class(  'person'               , 24 ,       11 , 'human'           , 6       , True         , False        , (220, 20, 60) ),
        Class(  'rider'                , 25 ,       12 , 'human'           , 6       , True         , False        , (255,  0,  0) ),
        Class(  'car'                  , 26 ,       13 , 'vehicle'         , 7       , True         , False        , (  0,  0,142) ),
        Class(  'truck'                , 27 ,       14 , 'vehicle'         , 7       , True         , False        , (  0,  0, 70) ),
        Class(  'bus'                  , 28 ,       15 , 'vehicle'         , 7       , True         , False        , (  0, 60,100) ),
        Class(  'caravan'              , 29 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0, 90) ),
        Class(  'trailer'              , 30 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0,110) ),
        Class(  'train'                , 31 ,       16 , 'vehicle'         , 7       , True         , False        , (  0, 80,100) ),
        Class(  'motorcycle'           , 32 ,       17 , 'vehicle'         , 7       , True         , False        , (  0,  0,230) ),
        Class(  'bicycle'              , 33 ,       18 , 'vehicle'         , 7       , True         , False        , (119, 11, 32) ),
        Class(  'license plate'        , -1 ,       -1 , 'vehicle'         , 7       , False        , True         , (  0,  0,142) ),
    ]
    carla_extra_classes = [
        Class(  'road line'            , 34 ,        0 , 'flat'            , 1       , False        , False         , (157,234, 50) ),
        Class(  'water'                , 35 ,      255 , 'void'            , 0       , False        , True          , ( 45, 60,150) ),
        Class(  'anomaly'              , 36 ,      255 , 'void'            , 0       , False        , True          , (255,255,255) ),
        Class(  'other'                , 37 ,      255 , 'void'            , 0       , False        , True          , ( 55, 90, 80) ),
        Class(  'static'               ,  4 ,      255 , 'void'            , 0       , False        , True          , (110,190,160) ),
        Class(  'dynamic'              ,  5 ,      255 , 'void'            , 0       , False        , True          , (170,120, 50) ),
        Class(  'fence'                , 13 ,        4 , 'construction'    , 2       , False        , False         , (100, 40, 40) ),
        Class(  'terrain'              , 22 ,        9 , 'nature'          , 4       , False        , False         , (145,170,100) ),
    ]
    # yapf: enable
    # Deduplicate according to id
    id2class = {i.id: i for i in classes}
    for i in carla_extra_classes:
        id2class[i.id] = i
    classes = [i for i in id2class.values()]

    color2id = {i.color: i.id for i in classes}

    return color2id


def prep_dst_folder(dst_path):
    os.makedirs(os.path.join(dst_path, "train", "semantic"), exist_ok=True)
    os.makedirs(os.path.join(dst_path, "train", "rgb"), exist_ok=True)
    os.makedirs(os.path.join(dst_path, "val", "semantic"), exist_ok=True)
    os.makedirs(os.path.join(dst_path, "val", "rgb"), exist_ok=True)


def carlacolorfile2labelfile(args):
    color_map_file, save_path = args
    color_map = np.asarray(Image.open(color_map_file))
    height, width, _ = color_map.shape
    label_map = np.ones(shape=(height, width), dtype=np.int16) * 250
    color2id = get_coder()
    for idx, (color, id) in enumerate(color2id.items()):
        mask = (color_map == color).all(axis=2) 
        label_map[mask] = id
    if (label_map == 250).any():
        color_map_ = np.copy(color_map)
        color_map_[label_map != 250] = (0,0,0)
        Image.fromarray(color_map_).save('/tmp/save.png')
        raise RuntimeError
    Image.fromarray(label_map).save(save_path)

import multiprocessing

def main():
    opts = get_argparser().parse_args()
    # assert opts.srs_dir is not None and opts.dataset_name is not None, "use --srs_dir and --dataset_name to specify source and destination folders"
    opts.dst_dir = os.path.join(opts.dst_dir, opts.dataset_name)
    prep_dst_folder(opts.dst_dir)


    if opts.from_file is None:
        srs_filenames = [name[:-8] for name in os.listdir(opts.srs_dir) if "semantic" not in name]
        filename_train, filename_val = train_test_split(srs_filenames, test_size=0.2, random_state=42)

        for file in tqdm(filename_train, desc="Processing train set"):
            color_map = np.array(Image.open(os.path.join(opts.srs_dir, f"{file}_semantic.png")))
            label_map = carlacolorfile2labelfile(color_map)
            Image.fromarray(label_map).save(os.path.join(opts.dst_dir, "train", "semantic", f"{file}_semantic.png"))
            if not opts.label_only:
                shutil.copyfile(os.path.join(opts.srs_dir, f"{file}_rgb.png"),
                                os.path.join(opts.dst_dir, "train", "rgb", f"{file}_rgb.png"))

        for file in tqdm(filename_val, desc="Processing val set"):
            color_map = np.array(Image.open(os.path.join(opts.srs_dir, f"{file}_semantic.png")))
            label_map = carlacolorfile2labelfile(color_map)
            Image.fromarray(label_map).save(os.path.join(opts.dst_dir, "val", "semantic", f"{file}_semantic.png"))
            if not opts.label_only:
                shutil.copyfile(os.path.join(opts.srs_dir, f"{file}_rgb.png"),
                                os.path.join(opts.dst_dir, "val", "rgb", f"{file}_rgb.png"))

    else:  #Town-12
        dirs_rgb, dirs_mask, scene_id = [], [], []
        with open(opts.from_file, "r") as f:
            filelines = f.readlines()
            for line in filelines:
                rgb_file, mask_file, _, _ = line.split(",")
                dirs_rgb.append(rgb_file)
                dirs_mask.append(mask_file)
                scene = rgb_file.split("/")[6]
                scene_id.append(scene)
        X_train, X_test, y_train, y_test, scene_train, scene_test = train_test_split(dirs_rgb,
                                                                                     dirs_mask,
                                                                                     scene_id,
                                                                                     test_size=0.2,
                                                                                     stratify=scene_id,
                                                                                     random_state=42)

        for file, scene in tqdm(zip(X_train, scene_train), desc="Processing X_train", total=len(scene_train)):
            os.symlink(os.path.abspath(file),
                       os.path.join(opts.dst_dir, "train", "rgb", f"{scene}_{file.split('/')[-1]}"))

        for file, scene in tqdm(zip(X_test, scene_test), desc="Processing X_test", total=len(scene_test)):
            os.symlink(os.path.abspath(file), os.path.join(opts.dst_dir, "val", "rgb",
                                                           f"{scene}_{file.split('/')[-1]}"))

        args = []
        for file, scene in tqdm(zip(y_train, scene_train), desc="Processing y_train", total=len(scene_train)):
            args.append((file, os.path.join(opts.dst_dir, "train", "semantic", f"{scene}_{file.split('/')[-1]}")))

        with multiprocessing.Pool(120) as p:
            for _ in tqdm(p.imap_unordered(carlacolorfile2labelfile, args), total=len(args)):
                pass

        print('finish1')

        args = []
        for file, scene in tqdm(zip(y_test, scene_test), desc="Processing y_test", total=len(scene_test)):
            args.append((file, os.path.join(opts.dst_dir, "val", "semantic", f"{scene}_{file.split('/')[-1]}")))

        with multiprocessing.Pool(120) as p:
            for _ in tqdm(p.imap_unordered(carlacolorfile2labelfile, args), total=len(args)):
                pass

        print('finish2')


if __name__ == "__main__":
    main()
