import glob
import torch
from torch.utils.data import DataLoader
from dataset.dataloader_semantic_THAB import SemanticKitti
from models.semanticFCN import SemanticNetworkWithFPN
from models.losses import TverskyLoss, SemanticSegmentationLoss
import torch.optim as optim
import tqdm
import time
import numpy as np
import cv2
import os
import open3d as o3d
from dataset.definitions import color_map, class_names
from torch.utils.tensorboard import SummaryWriter
import argparse
import json

from models.tester import Tester
from models.trainer import Trainer

def count_folders(directory):
    return len([name for name in os.listdir(directory) if os.path.isdir(os.path.join(directory, name))])


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



def main(args):
    config = {}
    config["BACKBONE"] = args.model_type
    config["USE_ATTENTION"] = args.attention
    config["USE_NORMALS"] = args.normals
    config["USE_MULTI_SCALE"] = args.multi_scale_meta
    config["USE_PRETRAINED"] = args.pretrained
    config["TEST_SCENE"] = args.test_id
    config["NUM_CLASSES"] = 20
    config["CLASS_NAMES"] = class_names
    config["CLASS_COLORS"] = color_map
    config["NUM_EPOCHS"] = args.num_epochs
    config["BATCH_SIZE"] = args.batch_size

    num_folder = count_folders("/home/appuser/data/SemanticTHAB/sequences/")

    if args.test_id == -1:
        data_path_train = [(bin_path, bin_path.replace("velodyne", "labels").replace("bin", "label")) for folder in [str(i).zfill(4) for i in range(num_folder)]  for bin_path in glob.glob(f"/home/appuser/data/SemanticTHAB/sequences/{folder}/velodyne/*.bin")]
        data_path_test = [(bin_path, bin_path.replace("velodyne", "labels").replace("bin", "label")) for folder in [str(i).zfill(4) for i in range(num_folder) if i == 2] for bin_path in glob.glob(f"/home/appuser/data/SemanticTHAB/sequences/{folder}/velodyne/*.bin")]
    else:
        data_path_train = [(bin_path, bin_path.replace("velodyne", "labels").replace("bin", "label")) for folder in [str(i).zfill(4) for i in range(num_folder) if i != args.test_id] for bin_path in glob.glob(f"/home/appuser/data/SemanticTHAB/sequences/{folder}/velodyne/*.bin")]
        data_path_test = [(bin_path, bin_path.replace("velodyne", "labels").replace("bin", "label")) for folder in [str(i).zfill(4) for i in range(num_folder) if i == args.test_id] for bin_path in glob.glob(f"/home/appuser/data/SemanticTHAB/sequences/{folder}/velodyne/*.bin")]
    
    data_path_train = data_path_train[0:100]
    data_path_test = data_path_train[0:100]

    depth_dataset_train = SemanticKitti(data_path_train, rotate=args.rotate, flip=args.flip)
    dataloader_train = DataLoader(depth_dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    
    depth_dataset_test = SemanticKitti(data_path_test, rotate=False, flip=False)
    dataloader_test = DataLoader(depth_dataset_test, batch_size=1, shuffle=False, num_workers=args.num_workers)
    
    # Depth Estimation Network
    if args.normals:
        model = SemanticNetworkWithFPN(backbone=args.model_type, meta_channel_dim=6, num_classes=20, attention=args.attention, multi_scale_meta=args.multi_scale_meta)
    else:
        model = SemanticNetworkWithFPN(backbone=args.model_type, meta_channel_dim=3, num_classes=20, attention=args.attention, multi_scale_meta=args.multi_scale_meta)

    num_params = count_parameters(model)
    config["NUM_PARAMS"] = num_params
    print("num_params", count_parameters(model))
    

    if args.pretrained:
        load_path ='/home/appuser/data/train_semantic_kitti/{}_{}{}{}/'.format(args.model_type, "a" if args.attention else "", "n" if args.normals else "", "m" if args.multi_scale_meta else "")
        config["PRETAINED_PATH"] = load_path
        model.load_state_dict(torch.load(os.path.join(load_path,"model_final.pth")))
    
    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = "min", factor = 0.1)
        
    # Save Path
    if args.test_id != -1:
        save_path_p1 ='/home/appuser/data/train_semantic_THAB_v2/test_split_{}/'.format(str(args.test_id).zfill(4))
    else:
        save_path_p1 ='/home/appuser/data/train_semantic_THAB_v2/test_split_{}/'.format("final")
    

    save_path_p2 ='{}_{}{}{}{}/'.format(args.model_type, "a" if args.attention else "", "n" if args.normals else "", "m" if args.multi_scale_meta else "", "p" if args.pretrained else "")
    save_path = os.path.join(save_path_p1,save_path_p2)
    os.makedirs(save_path, exist_ok=True)

    config["SAVE_PATH"] = save_path

    save_path = os.path.dirname(config["SAVE_PATH"])
    # write config file
    with open(os.path.join(save_path, "config.json"), 'w') as fp:
        json.dump(config, fp)

    # train model
    trainer = Trainer(model, optimizer, save_path, config, scheduler = scheduler, visualize = True)
    trainer(dataloader_train, dataloader_test, args.num_epochs)

    # test final model
    tester = Tester(model, save_path=os.path.join(save_path, "model_final.pth"), config=config, load=False)
    tester(dataloader_test)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Train script for SemanticTHAB')
    parser.add_argument('--model_type', type=str, default='resnet34',
                        help='Type of the model to be used (default: resnet34)')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate for the model (default: 0.001)')
    parser.add_argument('--num_epochs', type=int, default=5,
                        help='Number of epochs for training (default: 50)')
    parser.add_argument('--test_every_nth_epoch', type=int, default=1,
                        help='Test every nth epoch (default: 1)')
    parser.add_argument('--test_id', type=int, default=-1,
                        help='Test ID of the test sequence for the leave one out CV. -1 for training on all')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for training (default: 1)')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='Number of data loading workers (default: 1)')
    parser.add_argument('--rotate', action='store_true',
                        help='Whether to apply rotation augmentation (default: False)')
    parser.add_argument('--attention', action='store_true',
                        help='Whether to use attention (default: False)')
    parser.add_argument('--normals', action='store_true',
                        help='Whether to normals as input (default: False)')
    parser.add_argument('--multi_scale_meta', action='store_true',
                        help='Whether to to inject meta data at multiple scales (default: False)')
    parser.add_argument('--pretrained', action='store_true',
                        help='Whether to use a model pretrained on SemanticKitti (default: False)')
    parser.add_argument('--flip', action='store_true',
                        help='Whether to apply flip augmentation (default: False)')
    parser.add_argument('--visualization', action='store_true',
                        help='Toggle visualization during training (default: False)')
    args = parser.parse_args()

    main(args)
