from tqdm import tqdm
import network
import utils
import os
import random
import argparse
import numpy as np
from datetime import datetime
import wandb
import csv

from torch.utils import data
from datasets import *
from utils import ext_transforms as et
from metrics import StreamSegMetrics

import torch
import torch.nn as nn
from utils.visualizer import Visualizer

from PIL import Image
import matplotlib
import matplotlib.pyplot as plt


def get_argparser():
    parser = argparse.ArgumentParser()

    # Added Options
    parser.add_argument("--run_name", type=str, default='unnamed',
                        help="Custom name of the run")
    parser.add_argument("--wandb", action='store_true', default=False,
                        help='Inject W&B monitoring')
    parser.add_argument("--acdc_scene", type=str, default=None, choices=['snow', 'rain', 'fog', 'night'],
                        help="Set ACDC dataset scenario")
    parser.add_argument("--coder", type=str, choices=['voc', 'cityscapes', 'nightlab', 'carla'], default=None,
                        help='Select train_id mapper, now depreciated.')
    parser.add_argument("--boost_dataset", type=str, default=None,
                        choices=['carla'], help='Name of dataset for boosting')
    parser.add_argument("--boost_data_root", type=str, default=None,
                        help="path to dataset for boosting")
    parser.add_argument("--boost_strength", type=float, default=0.5,
                        help='The ratio of boost dataset per batch. boost_data_batch_size = round(batch_size * boost_strength). real_data_batch_size = batch_size - boost_data_batch_size')
    parser.add_argument("--disable_fci", action='store_true', default=False,
                        help='When using boost dataset, Force Complete Iteration (FCI) will ensure that no data is left unseen by the model. Use this option to disable FCI so that each epoch stops when one of the datasets finished iterating')
    parser.add_argument("--class_weights", type=str, default=None,
                        help="Path to the weight for loss calculation, default to None")
    parser.add_argument("--subsample", type=float, default=None, 
                        help="Take a random subsample of the full dataset, float input. Default is None")
    parser.add_argument("--freeze_bn", action="store_true", default=False)
    parser.add_argument("--mix", type=str, choices=['blend', 'concat', 'boost'], default='concat')
    parser.add_argument("--mix_masterinput", type=str, default="0.5,0.5")
    parser.add_argument("--sample_stat", type=str, default=None)
    parser.add_argument("--sample_sum", type=str, default=None)
    parser.add_argument("--class_sample_mask", type=str, default=None)
    parser.add_argument("--blend_schedule", type=str, default=None, help="Set blending schedule. Format: \"period,increment,reset_lr\". Example: \"3000,0.1,1\" means increment blending ratio by 0.1 every 3000 iterations, reset lr each time.")
    # Datset Options
    parser.add_argument("--data_root", type=str, nargs='+', required=True,
                        help="path to Dataset. Could input multiple data roots. Note that the number of dataset arguments must equate the number of data_root arguments.")
    parser.add_argument("--dataset", type=str, nargs='+', required=True,
                        choices=['voc', 'cityscapes', 'nightlab', 'carla', 'bdd-100k', 'acdc', 'sampled_carla', 'sampled_carla_masked'], help='Name of dataset')
    parser.add_argument("--num_classes", type=int, default=None,
                        help="num classes (default: None)")

    # Deeplab Options
    available_models = sorted(name for name in network.modeling.__dict__ if name.islower() and \
                              not (name.startswith("__") or name.startswith('_')) and callable(
                              network.modeling.__dict__[name])
                              )
    parser.add_argument("--model", type=str, default='deeplabv3plus_resnet101',
                        choices=available_models, help='model name')
    parser.add_argument("--separable_conv", action='store_true', default=False,
                        help="apply separable conv to decoder and aspp")
    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])

    # Train Options
    parser.add_argument("--test_only", action='store_true', default=False)
    parser.add_argument("--save_val_results", type=int, default=None,
                        help="save segmentation results to \"./results\"")
    parser.add_argument("--total_itrs", type=int, default=30e3,
                        help="epoch number (default: 30k)")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="learning rate (default: 0.01)")
    parser.add_argument("--lr_policy", type=str, default='poly', choices=['poly', 'step'],
                        help="learning rate scheduler policy")
    parser.add_argument("--step_size", type=int, default=10000)
    parser.add_argument("--crop_val", action='store_true', default=False,
                        help='crop validation (default: False)')
    parser.add_argument("--batch_size", type=int, default=16,
                        help='batch size (default: 16)')
    parser.add_argument("--val_batch_size", type=int, default=4,
                        help='batch size for validation (default: 4)')
    parser.add_argument("--crop_size", type=int, default=768)

    parser.add_argument("--ckpt", default=None, type=str,
                        help="restore from checkpoint")
    parser.add_argument("--continue_training", action='store_true', default=False)

    parser.add_argument("--loss_type", type=str, default='cross_entropy',
                        choices=['cross_entropy', 'focal_loss'], help="loss type (default: False)")
    parser.add_argument("--gpu_id", type=str, default='0',
                        help="GPU ID")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help='weight decay (default: 1e-4)')
    parser.add_argument("--random_seed", type=int, default=1,
                        help="random seed (default: 1)")
    parser.add_argument("--print_interval", type=int, default=10,
                        help="print interval of loss (default: 10)")
    parser.add_argument("--val_interval", type=int, default=100,
                        help="epoch interval for eval (default: 100)")
    parser.add_argument("--download", action='store_true', default=False,
                        help="download datasets")
    parser.add_argument("--debug", action='store_true', default=False,
                        help="debug mode")

    # PASCAL VOC Options
    parser.add_argument("--year", type=str, default='2012',
                        choices=['2012_aug', '2012', '2011', '2009', '2008', '2007'], help='year of VOC')

    # Visdom options
    parser.add_argument("--enable_vis", action='store_true', default=False,
                        help="use visdom for visualization")
    parser.add_argument("--vis_port", type=str, default='13570',
                        help='port for visdom')
    parser.add_argument("--vis_env", type=str, default='main',
                        help='env for visdom')
    parser.add_argument("--vis_num_samples", type=int, default=8,
                        help='number of samples for visualization (default: 8)')
    return parser


def get_dataset(opts):
    """ Dataset And Augmentation
    """
    train_dst_list = []
    val_dst_list = []
    for dataset, root in zip(opts.dataset, opts.data_root):
        if dataset == 'voc':
            train_transform = et.ExtCompose([
                # et.ExtResize(size=opts.crop_size),
                et.ExtRandomScale((0.5, 2.0)),
                et.ExtRandomCrop(size=(opts.crop_size, opts.crop_size), pad_if_needed=True),
                et.ExtRandomHorizontalFlip(),
                et.ExtToTensor(),
                et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
            ])
            if opts.crop_val:
                val_transform = et.ExtCompose([
                    et.ExtResize(opts.crop_size),
                    et.ExtCenterCrop(opts.crop_size),
                    et.ExtToTensor(),
                    et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]),
                ])
            else:
                val_transform = et.ExtCompose([
                    et.ExtToTensor(),
                    et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]),
                ])
        else:
            train_transform = et.ExtCompose([
                # et.ExtResize( 512 ),
                et.ExtRandomCrop(size=(opts.crop_size, opts.crop_size)),
                et.ExtColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
                et.ExtRandomHorizontalFlip(),
                et.ExtToTensor(),
                et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
            ])

            val_transform = et.ExtCompose([
                # et.ExtResize(opts.crop_size),
                # et.ExtCenterCrop(opts.crop_size),
                et.ExtToTensor(),
                et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
            ])

        if dataset == 'voc':
            train_dst = VOCSegmentation(root=root, year=opts.year,
                                        image_set='train', download=opts.download, transform=train_transform)
            val_dst = VOCSegmentation(root=root, year=opts.year,
                                    image_set='val', download=False, transform=val_transform)

        elif dataset == 'cityscapes':
            train_dst = Cityscapes(root=root,
                                split='train', coder=opts.coder, transform=train_transform, subsample=opts.subsample)
            val_dst = Cityscapes(root=root,
                                split='val', coder=opts.coder, transform=val_transform)
            
        elif dataset == 'nightlab':
            train_dst = NightLab(root=root,
                                split='train', coder=opts.coder, transform=train_transform)
            val_dst = NightLab(root=root,
                                split='val', coder=opts.coder, transform=val_transform)
            
        elif dataset == 'bdd-100k':
            train_dst = BDD_100K(root=root,
                                split='train', coder=opts.coder, transform=train_transform)
            val_dst = BDD_100K(root=root,
                                split='val', coder=opts.coder, transform=val_transform)
            
        elif dataset == 'acdc':
            assert opts.acdc_scene is not None, "set ACDC scene with --acdc_scene"

            train_dst = ACDC(root=root, scene=opts.acdc_scene,
                                split='train', coder=opts.coder, transform=train_transform)
            val_dst = ACDC(root=root, scene=opts.acdc_scene,
                                split='val', coder=opts.coder, transform=val_transform)
            
        elif dataset == 'carla':
            train_dst = Carla(root=root,
                                split='train', transform=train_transform)
            val_dst = Carla(root=root,
                                split='val', transform=val_transform)

        elif dataset == 'sampled_carla':
            train_dst = SampledCarlaDataset(root,
            'train', opts.sample_stat, opts.sample_sum, custom_mask=opts.class_sample_mask, transform=train_transform)
            val_dst = Carla(root,
                                split='val', transform=val_transform)
        elif dataset == 'sampled_carla_masked':
            train_dst = SampledCarlaMaskDataset(root,
            'train', opts.sample_stat, opts.sample_sum, custom_mask=opts.class_sample_mask, transform=train_transform)
            val_dst = Carla(root,
                                split='val', transform=val_transform)
            
        train_dst_list.append(train_dst)
        val_dst_list.append(val_dst)

    if len(train_dst_list) == 1:
        train_dst = train_dst_list[0]
        val_dst = val_dst_list[0]
    else:
        if opts.mix == "concat":
            train_dst = torch.utils.data.ConcatDataset(train_dst_list)
            train_dst.decode_target = train_dst_list[0].decode_target
            val_dst = torch.utils.data.ConcatDataset(val_dst_list)
            val_dst.decode_target = val_dst_list[0].decode_target
        elif opts.mix == "blend":
            sample_prob = [float(i) for i in opts.mix_masterinput.split(",")]
            train_dst = BlendedDataset(sample_prob, *train_dst_list)
            train_dst.decode_target = train_dst_list[0].decode_target
            val_dst = val_dst_list[0]
        elif opts.mix == "boost":
            raise NotImplementedError
    
    if opts.boost_dataset is not None:
        if opts.boost_dataset.lower() == 'carla':
            boost_dst = Carla(root=opts.boost_data_root,
                              split='both', transform=train_transform)
            if opts.coder is not None:
                if opts.coder.lower() == 'voc':
                    boost_dst.decode_target = VOCSegmentation.decode_target
                elif opts.coder.lower() == 'cityscapes':
                    boost_dst.decode_target = Cityscapes.decode_target
                elif opts.coder.lower() == 'nightlab':
                    boost_dst.decode_target = NightLab.decode_target
                elif opts.coder.lower() == 'carla':
                    boost_dst.decode_target = Carla.decode_target
    
    # change the decoder if specified
    if opts.coder is not None:
        if opts.coder.lower() == 'voc':
            train_dst.decode_target = VOCSegmentation.decode_target
            val_dst.decode_target = VOCSegmentation.decode_target
        elif opts.coder.lower() == 'cityscapes':
            train_dst.decode_target = Cityscapes.decode_target
            val_dst.decode_target = Cityscapes.decode_target
        elif opts.coder.lower() == 'nightlab':
            train_dst.decode_target = NightLab.decode_target
            val_dst.decode_target = NightLab.decode_target
        elif opts.coder.lower() == 'carla':
            train_dst.decode_target = Carla.decode_target
            val_dst.decode_target = Carla.decode_target
            
    if opts.boost_dataset is not None:
        return train_dst, val_dst, boost_dst
    else:
        return train_dst, val_dst


def validate(opts, model, loader, device, metrics, iter, criterion=None, ret_samples_ids=None):
    """Do validation and return specified samples"""
    metrics.reset()
    ret_samples = []
    img_to_store = opts.save_val_results if opts.save_val_results is not None else 0
    if opts.save_val_results is not None:
        utils.mkdir("results")
        utils.mkdir(f"results/{opts.run_name}")
        utils.mkdir(f"results/{opts.run_name}/val_samples")
        denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
        img_id = 0
        
    val_loss = []
    
    iterator = loader.__iter__()
    
    with torch.no_grad():
        for i in tqdm(range(len(iterator))):
            images, labels = iterator.__next__()
            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            outputs = model(images)
            
            if criterion is not None and opts.wandb:
                loss = criterion(outputs, labels)
                val_loss.append(loss.detach().cpu().numpy())
            
            preds = outputs.detach().max(dim=1)[1].cpu().numpy()
            targets = labels.cpu().numpy()

            metrics.update(targets, preds)
            if ret_samples_ids is not None and i in ret_samples_ids:  # get vis samples
                ret_samples.append(
                    (images[0].detach().cpu().numpy(), targets[0], preds[0]))

            if img_to_store != 0:
                image = images[0].detach().cpu().numpy()
                target = targets[0]
                pred = preds[0]

                image = (denorm(image) * 255).transpose(1, 2, 0).astype(np.uint8)
                target = loader.dataset.decode_target(target).astype(np.uint8)
                pred = loader.dataset.decode_target(pred).astype(np.uint8)

                Image.fromarray(image).save(f'results/{opts.run_name}/val_samples/{img_id}_image.png')
                Image.fromarray(target).save(f'results/{opts.run_name}/val_samples/{img_id}_target.png')
                Image.fromarray(pred).save(f'results/{opts.run_name}/val_samples/{img_id}_pred_iter{iter}.png')

                fig = plt.figure()
                plt.imshow(image)
                plt.axis('off')
                plt.imshow(pred, alpha=0.7)
                ax = plt.gca()
                ax.xaxis.set_major_locator(matplotlib.ticker.NullLocator())
                ax.yaxis.set_major_locator(matplotlib.ticker.NullLocator())
                plt.savefig(f'results/{opts.run_name}/val_samples/{img_id}_overlay_iter{iter}.png', bbox_inches='tight', pad_inches=0)
                plt.close()
                
                img_id += 1
                img_to_store -= 1

        score = metrics.get_results()
        
        if opts.test_only:
            _class_iou = list(score["Class IoU"].values())
            _run_name = opts.ckpt.split('/')[-2]
            with open(f"checkpoints/{_run_name}/class_iou.csv", "w") as f:
                write = csv.writer(f)
                write.writerow(_class_iou)
        # print(score["Class IoU"])
        # print(type(score["Class IoU"]))
        
        if opts.wandb:
            wandb.log({"val_loss": np.average(val_loss),
                       "val_mIoU": score["Mean IoU"],
                       "class_IoU": score["Class IoU"]}, step=iter)
    return score, ret_samples


def main():
    opts = get_argparser().parse_args()
    assert len(opts.dataset) == len(opts.data_root), "Specify the path of dataset with --data_root for each dataset"

    opts.run_name = f"{opts.run_name}_{datetime.now().strftime('%Y-%m-%d_%H%M%S')}"
    
    if opts.disable_fci:
        assert opts.boost_dataset, "This option can only be used when boosting with another dataset"
    
    if opts.coder is not None:
        if opts.coder.lower() == 'voc':
            opts.num_classes = 21
        elif opts.coder.lower() == 'cityscapes':
            opts.num_classes = 19
        elif opts.coder.lower() == 'nightlab':
            opts.num_classes = 19
        elif opts.coder.lower() == 'carla':
            opts.num_classes = 19
    else:
        if opts.dataset[0].lower() == 'voc':
            opts.num_classes = 21
        else:
            opts.num_classes = 19
        # elif opts.dataset[0].lower() == 'cityscapes':
        #     opts.num_classes = 19
        # elif opts.dataset[0].lower() == 'nightlab':
        #     opts.num_classes = 19
        # elif opts.dataset[0].lower() == 'bdd-100k':
        #     opts.num_classes = 19
        # elif opts.dataset[0].lower() == 'acdc':
        #     opts.num_classes = 19
        # elif opts.dataset[0].lower() == 'carla':
        #     opts.num_classes = 19
    
    # Setup visualization
    vis = Visualizer(port=opts.vis_port,
                     env=opts.vis_env) if opts.enable_vis else None
    if vis is not None:  # display options
        vis.vis_table("Options", vars(opts))

    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device)

    # Setup random seed
    torch.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)
    
    # Setup dataloader
    if opts.dataset[0] == 'voc' and not opts.crop_val:
        opts.val_batch_size = 1

    if opts.boost_dataset is not None:
        assert 0 < opts.boost_strength <= 1, "--boost_strength must be > 0 & <= 1"
        opts.boost_batch_size = round(opts.batch_size * opts.boost_strength)
        opts.batch_size = opts.batch_size - opts.boost_batch_size
        
        train_dst, val_dst, boost_dst = get_dataset(opts)

        boost_loader = data.DataLoader(boost_dst, batch_size=opts.boost_batch_size, shuffle=True, num_workers=2, drop_last=True)
        print(f"Total batch size: {opts.batch_size + opts.boost_batch_size}\nReal batch size: {opts.batch_size}\nBoost batch size: {opts.boost_batch_size}")
        print(f"Boost set: {len(boost_dst)}")
    else:
        train_dst, val_dst = get_dataset(opts)
    
    if opts.wandb:
        wandb.init(project="DeepLabv3plus",
                   name=opts.run_name,
                   config=opts)
    
    if opts.batch_size == 0:
        train_loader = boost_loader
        opts.boost_dataset = None
    else:
        if opts.mix == "blend" or "sampled_carla" in opts.dataset or "sampled_carla_masked" in opts.dataset:
            train_loader = data.DataLoader(train_dst, batch_size=opts.batch_size, num_workers=2,drop_last=True)  # drop_last=True to ignore single-image batches.
        else:
            train_loader = data.DataLoader(train_dst, batch_size=opts.batch_size, shuffle=True, num_workers=2,drop_last=True)  # drop_last=True to ignore single-image batches.

    val_loader = data.DataLoader(val_dst, batch_size=opts.val_batch_size, shuffle=False, num_workers=2)

    if opts.mix == "blend":
        print(f"Dataset:\t{', '.join(opts.dataset)}\nMix:\t Blend\nBlend Probability:\t{opts.mix_masterinput}")
    elif "sampled_carla" in opts.dataset or "sampled_carla_masked" in opts.dataset:
        print(f"Dataset:\t{', '.join(opts.dataset)}")
    else:
        print("Dataset: %s, Train set: %d, Val set: %d" %(opts.dataset, len(train_dst), len(val_dst)))

    # Set up model (all models are 'constructed at network.modeling)
    model = network.modeling.__dict__[opts.model](num_classes=opts.num_classes, output_stride=opts.output_stride)
    if opts.separable_conv and 'plus' in opts.model:
        network.convert_to_separable_conv(model.classifier)
    utils.set_bn_momentum(model.backbone, momentum=0.01)
    
    # Set up metrics
    metrics = StreamSegMetrics(opts.num_classes)

    # Set up optimizer
    optimizer = torch.optim.SGD(params=[
        {'params': model.backbone.parameters(), 'lr': 0.1 * opts.lr},
        {'params': model.classifier.parameters(), 'lr': opts.lr},
    ], lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)
    # optimizer = torch.optim.SGD(params=model.parameters(), lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)
    # torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.lr_decay_step, gamma=opts.lr_decay_factor)
    if opts.lr_policy == 'poly':
        scheduler = utils.PolyLR(optimizer, opts.total_itrs, power=0.9)
    elif opts.lr_policy == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.step_size, gamma=0.1)

    if opts.class_weights:
        class_weights = torch.Tensor(np.load(opts.class_weights)).to(device)
    else:
        class_weights = None
        
    # Set up criterion
    # criterion = utils.get_loss(opts.loss_type)
    if opts.loss_type == 'focal_loss':
        criterion = utils.FocalLoss(ignore_index=255, size_average=True)
    elif opts.loss_type == 'cross_entropy':
        criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='mean', weight=class_weights)

    def save_ckpt(path):
        """ save current model
        """
        torch.save({
            "cur_itrs": cur_itrs,
            "model_state": model.module.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_score": best_score,
        }, path)
        print("Model saved as %s" % path)
    
    utils.mkdir('checkpoints')
    # Restore
    best_score = 0.0
    cur_itrs = 0
    cur_epochs = 0
    if opts.ckpt is not None and os.path.isfile(opts.ckpt):
        # https://github.com/VainF/DeepLabV3Plus-Pytorch/issues/8#issuecomment-605601402, @PytaichukBohdan
        checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'))
        try:
            model.load_state_dict(checkpoint["model_state"])
        except RuntimeError:
            utils.fix_bn(model.backbone, affine=True)
            model.load_state_dict(checkpoint["model_state"])
        model = nn.DataParallel(model)
        model.to(device)
        if opts.continue_training:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            cur_itrs = checkpoint["cur_itrs"]
            best_score = checkpoint['best_score']
            print("Training state restored from %s" % opts.ckpt)
        print("Model restored from %s" % opts.ckpt)
        del checkpoint  # free memory
    else:
        print("[!] Retrain")
        model = nn.DataParallel(model)
        model.to(device)
 
    if opts.freeze_bn:
        utils.fix_bn(model.module.backbone, affine=True)
    
    # ==========   Train Loop   ==========#
    vis_sample_id = np.random.randint(0, len(val_loader), opts.vis_num_samples,
                                      np.int32) if opts.enable_vis else None  # sample idxs for visualization
    denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # denormalization for ori images

    if opts.test_only:
        model.eval()
        val_score, ret_samples = validate(
            opts=opts, model=model, loader=val_loader, device=device, metrics=metrics, iter="_test", ret_samples_ids=vis_sample_id)
        print(metrics.to_str(val_score))
        return

    utils.mkdir(f"checkpoints/{opts.run_name}")
    
    best_ckpt_filename = ""
    interval_loss = 0
    
    if opts.blend_schedule:
        assert opts.mix == "blend"
        schedule_p, blend_increment = opts.blend_schedule.split(",")
        schedule_p = int(schedule_p)
        blend_increment = float(blend_increment)
        sample_prob = [float(i) for i in opts.mix_masterinput.split(",")]
    
    while True:  # cur_itrs < opts.total_itrs:
        # =====  Train  =====
        model.train()
        
        # if opts.freeze_bn:
        #     utils.fix_bn(model.module.backbone, affine=True)  # add .module since model is a DataParallel wrapper of actual model
        
        cur_epochs += 1
        
        train_iterator= train_loader.__iter__()
        if opts.boost_dataset is not None:
            boost_iterator = boost_loader.__iter__()
            
        train_loader_dry = False
        boost_loader_dry = False
        
        while True:
            cur_itrs += 1
            
            if opts.blend_schedule:
                if cur_itrs % schedule_p == 0 and sample_prob[0] != 1:
                    sample_prob[0] = sample_prob[0] + blend_increment if sample_prob[0] + blend_increment < 1 else 1.0
                    for n, i in enumerate(sample_prob[1:]):
                        sample_prob[n + 1] = i - blend_increment if i - blend_increment > 0 else 0.0
                    opts.mix_masterinput = ','.join([str(i) for i in sample_prob])
                    print("############################################################\n")
                    print(f"Blend ratio incremented.\nDataset:\t{', '.join(opts.dataset)}\nMix:\t Blend\nBlend Probability:\t{opts.mix_masterinput}")
                    print("\n############################################################")
                    train_dst, _ = get_dataset(opts)
                    train_loader = data.DataLoader(train_dst, batch_size=opts.batch_size, num_workers=2,drop_last=True)
                    train_iterator = train_loader.__iter__()
                
            try:
                images, labels = train_iterator.__next__()
                
            except StopIteration:
                if opts.disable_fci:
                    print("################ Authentic dataset depleted, starting new epoch ################")
                    break
                else:
                    train_loader_dry = True
                    if train_loader_dry and boost_loader_dry:
                        print("################ Both dataset depleted, starting new epoch ################")
                        break
                    else:
                        print("################ Authentic dataset depleted, reshuffling ################")
                        train_iterator = train_loader.__iter__()
                        images, labels = train_iterator.__next__()
                        cur_epochs += 1
                    
    
            if opts.boost_dataset is not None:
                try:
                    boost_images, boost_labels = boost_iterator.__next__()
                    
                except StopIteration:
                    if opts.disable_fci:
                        print("################ Boost dataset depleted, starting new epoch ################")
                        break
                    else:
                        boost_loader_dry = True
                        if train_loader_dry and boost_loader_dry:
                            print("################ Both dataset depleted, starting new epoch ################")
                            break
                        else:
                            print("################ Boost dataset depleted, reshuffling ################")
                            boost_iterator = boost_loader.__iter__()
                            boost_images, boost_labels = boost_iterator.__next__()

                images = torch.cat((images, boost_images), dim=0)
                labels = torch.cat((labels, boost_labels), dim=0)
            
            if opts.debug:
                denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
                
                utils.mkdir("debug")
                utils.mkdir(f"debug/{opts.run_name}")
                print(images.shape)
                print(labels.shape)
                for i in range(images.shape[0]):
                    fig = plt.figure(frameon=False)
                    ax = plt.Axes(fig, [0., 0., 1., 1.])
                    ax.set_axis_off()
                    fig.add_axes(ax)
                    ax.imshow(denorm(images[i].permute(1, 2, 0).numpy()), aspect='auto')
                    fig.savefig(f"debug/{opts.run_name}/{i}_rgb.png")
                    plt.close()
                    
                    fig = plt.figure(frameon=False)
                    ax = plt.Axes(fig, [0., 0., 1., 1.])
                    ax.set_axis_off()
                    fig.add_axes(ax)
                    ax.imshow(labels[i].numpy(), aspect='auto')
                    fig.savefig(f"debug/{opts.run_name}/{i}_semantic.png")
                    plt.close()
                return
                
            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            np_loss = loss.detach().cpu().numpy()
            interval_loss += np_loss
            if vis is not None:
                vis.vis_scalar('Loss', cur_itrs, np_loss)

            if (cur_itrs) % 10 == 0:
                interval_loss = interval_loss / 10
                print("Epoch %d, Itrs %d/%d, Loss=%f" %
                      (cur_epochs, cur_itrs, opts.total_itrs, interval_loss))
                
                if opts.wandb:
                    wandb.log({"train_loss": interval_loss}, step=cur_itrs)
                    
                interval_loss = 0.0

            if (cur_itrs) % opts.val_interval == 0:
                save_ckpt(f'checkpoints/{opts.run_name}/latest_{opts.model}_{opts.dataset[0]}_os{opts.output_stride}.pth')
                print("validation...")
                model.eval()
                val_score, ret_samples = validate(
                    opts=opts, model=model, criterion=criterion, loader=val_loader, device=device, metrics=metrics, iter=cur_itrs,
                    ret_samples_ids=vis_sample_id)
                print(metrics.to_str(val_score))
                if val_score['Mean IoU'] > best_score:  # save best model
                    if best_ckpt_filename != "":
                        os.remove(f"checkpoints/{opts.run_name}/{best_ckpt_filename}")
                    best_score = val_score['Mean IoU']
                    best_ckpt_filename = f"best_{best_score:.6f}_iter{cur_itrs}_{opts.model}_{opts.dataset[0]}_os{opts.output_stride}.pth"
                    save_ckpt(f'checkpoints/{opts.run_name}/{best_ckpt_filename}')

                if vis is not None:  # visualize validation score and samples
                    vis.vis_scalar("[Val] Overall Acc", cur_itrs, val_score['Overall Acc'])
                    vis.vis_scalar("[Val] Mean IoU", cur_itrs, val_score['Mean IoU'])
                    vis.vis_table("[Val] Class IoU", val_score['Class IoU'])

                    for k, (img, target, lbl) in enumerate(ret_samples):
                        img = (denorm(img) * 255).astype(np.uint8)
                        target = train_dst.decode_target(target).transpose(2, 0, 1).astype(np.uint8)
                        lbl = train_dst.decode_target(lbl).transpose(2, 0, 1).astype(np.uint8)
                        concat_img = np.concatenate((img, target, lbl), axis=2)  # concat along width
                        vis.vis_image('Sample %d' % k, concat_img)
                model.train()
            if opts.wandb:
                last_lr = scheduler.get_last_lr()
                wandb.log({"backbone_lr": last_lr[0], "classifier_lr": last_lr[1]}, step=cur_itrs)
            scheduler.step()

            if cur_itrs >= opts.total_itrs:
                return


if __name__ == '__main__':
    main()