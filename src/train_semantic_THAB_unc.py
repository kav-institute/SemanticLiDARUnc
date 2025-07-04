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
import yaml
import sys

from models.tester import Tester
from models.trainer import Trainer

def count_folders(directory):
    return len([name for name in os.listdir(directory) if os.path.isdir(os.path.join(directory, name))])


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def load_config(path):
    try:
        with open(path, 'r') as f:
            cfg = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"ERROR: config file not found!: {path}", file=sys.stderr)
        sys.exit(1)
    except yaml.YAMLError as exc:
        print(f"Error while parsing YAML-file: {exc}", file=sys.stderr)
        sys.exit(1)

    # Typcheck: wir erwarten z.B. ein Dict
    if not isinstance(cfg, dict):
        print(f"ERROR: Invalid format in {path} - expected Dict, found {type(cfg).__name__}", file=sys.stderr)
        sys.exit(1)

    return cfg

def main(args):
    cfg = load_config(args.cfg_path)

    # add additional configurations to config file
    cfg["extras"] = dict()
    cfg["extras"]["use_reflectivity"] = True
    cfg["extras"]["num_classes"] = 20
    cfg["extras"]["class_names"] = class_names
    cfg["extras"]["class_colors"] = color_map
    cfg["extras"]["loss_function"] = "Dirichlet"
    cfg["extras"]["with_calibration_loss"] = False  # only in combination with Dirichlet loss_function

    num_folder = count_folders(cfg["dataset_dir"])

    if cfg["logging_settings"]["test_id"] == -1:
        data_path_train = [(bin_path, bin_path.replace("velodyne", "labels").replace("bin", "label")) for folder in [str(i).zfill(4) for i in range(num_folder)]  for bin_path in glob.glob(f"{cfg['dataset_dir']}/{folder}/velodyne/*.bin")]
        data_path_test = [(bin_path, bin_path.replace("velodyne", "labels").replace("bin", "label")) for folder in [str(i).zfill(4) for i in range(num_folder) if i == 2] for bin_path in glob.glob(f"{cfg['dataset_dir']}/{folder}/velodyne/*.bin")]
    else:
        data_path_train = [(bin_path, bin_path.replace("velodyne", "labels").replace("bin", "label")) for folder in [str(i).zfill(4) for i in range(num_folder) if i != cfg["logging_settings"]["test_id"]] for bin_path in glob.glob(f"{cfg['dataset_dir']}/{folder}/velodyne/*.bin")]
        data_path_test = [(bin_path, bin_path.replace("velodyne", "labels").replace("bin", "label")) for folder in [str(i).zfill(4) for i in range(num_folder) if i == cfg["logging_settings"]["test_id"]] for bin_path in glob.glob(f"{cfg['dataset_dir']}/{folder}/velodyne/*.bin")]
    
    depth_dataset_train = SemanticKitti(data_path_train, rotate=cfg["model_settings"]["rotate"], flip=cfg["model_settings"]["flip"])
    dataloader_train = DataLoader(depth_dataset_train, batch_size=cfg["train_params"]["batch_size"], shuffle=True, num_workers=cfg["train_params"]["num_workers"])
    
    depth_dataset_test = SemanticKitti(data_path_test, rotate=False, flip=False)
    dataloader_test = DataLoader(depth_dataset_test, batch_size=1, shuffle=False, num_workers=cfg["train_params"]["num_workers"])
    
    # Depth Estimation Network
    if cfg["extras"]["use_reflectivity"]:
        input_channels = 2
    else:
        input_channels = 1
        
    # Semantic Segmentation Network
    if cfg["model_settings"]["normals"]:
        model = SemanticNetworkWithFPN(backbone=cfg["model_settings"]["model_type"], input_channels=input_channels, meta_channel_dim=6, num_classes=cfg["extras"]["num_classes"] , attention=cfg["model_settings"]["attention"], multi_scale_meta=cfg["model_settings"]["multi_scale_meta"])
    else:
        model = SemanticNetworkWithFPN(backbone=cfg["model_settings"]["model_type"], input_channels=input_channels, meta_channel_dim=3, num_classes=cfg["extras"]["num_classes"] , attention=cfg["model_settings"]["attention"], multi_scale_meta=cfg["model_settings"]["multi_scale_meta"])

    # count number of model parameters
    num_params = count_parameters(model)
    print("num_params", num_params)
    cfg["extras"]["num_params"] = num_params
    
    if os.path.isfile(cfg["model_settings"]["pretrained"]):
        try:
            model.load_state_dict(torch.load(cfg["model_settings"]["pretrained"]))
        except:
            print("WARNING: No pretrained model found")
    
    # Define optimizer
    optimizer = optim.AdamW(model.parameters(), lr=cfg["train_params"]["learning_rate"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = "min", factor = 0.1)
    
    if args.with_logging:
        # Save Path
        if cfg["logging_settings"]["test_id"] != -1:
            save_path_p1 =f"{cfg['logging_settings']['log_dir']}/home/devuser/workspace/data/train_semantic_THAB_v3/test_split_{str(cfg['logging_settings']['test_id']).zfill(4)}"
        else:
            save_path_p1 ="/home/devuser/workspace/data/train_semantic_THAB_v3/test_split_final"
        
        save_path_p2 ='{}_{}{}{}{}{}'.format(cfg["model_settings"]["model_type"], "a" if cfg["model_settings"]["attention"] else "", "n" if cfg["model_settings"]["normals"] else "", "m" if cfg["model_settings"]["multi_scale_meta"] else "", "p" if cfg["model_settings"]["pretrained"] else "", cfg["extras"]["loss_function"])
        save_path = os.path.join(save_path_p1,save_path_p2)
        
        # add time information
        t = time.gmtime()
        time_start = time.strftime("%y-%m-%d_%H-%M-%S", t)  # Changed format to avoid colons
        save_path = os.path.join(save_path, time_start)
        
        os.makedirs(save_path, exist_ok=True)
        cfg["extras"]["save_path"] = save_path

        #save_path = os.path.dirname(cfg["extras"]["save_path"])
        # write config file
        with open(os.path.join(cfg["extras"]["save_path"], "config.yaml"), "w") as file:
            yaml.safe_dump(
                cfg, 
                file,
                default_flow_style=False,   # use block style (indented) rather than inline when set to False
                sort_keys=False             # preserve the order in your dict (if PyYAML >=5.1) when set to False
            )
            
        # with open(os.path.join(cfg["extras"]["save_path"], "config.json"), 'w') as fp:
        #     json.dump(cfg, fp)

    # train model
    trainer = Trainer(model, optimizer, cfg, scheduler = scheduler, visualize = args.visualization, logging=args.with_logging)
    trainer(dataloader_train, dataloader_test)
    #cfg["train_params"]["num_epochs"], test_every_nth_epoch=cfg["logging_settings"]["test_every_nth_epoch"], save_every_nth_epoch=cfg["logging_settings"]["save_every_nth_epoch"])

    # test final model
    if args.with_logging:
        # with open(os.path.join(save_path, "config.json")) as json_data:
        #     config = json.load(json_data)
        #     json_data.close()
        
        test_mask = [0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1]

        #data_path_test = [(bin_path, bin_path.replace("velodyne", "labels").replace("bin", "label")) for bin_path in glob.glob(f"/home/appuser/data/SemanticTHAB/sequences/0006/velodyne/*.bin")]
        depth_dataset_test = SemanticKitti(data_path_test, rotate=False, flip=False)
        dataloader_test = DataLoader(depth_dataset_test, batch_size=1, shuffle=False, num_workers=4)

        #save_path = config["SAVE_PATH"]
        # test final model
        tester = Tester(model, save_path=os.path.join(cfg["extras"]["save_path"], "model_final.pt"), cfg=cfg, load=True, visualize=args.visualization, test_mask=test_mask)
        tester(dataloader_test)

if __name__ == '__main__':
    # general
    parser = argparse.ArgumentParser(description = 'Train script for SemanticTHAB')
    parser.add_argument('--visualization', 
                        #action='store_true',
                        type=bool, 
                        default=True,
                        help='Toggle visualization during training (default: False)')
    parser.add_argument('--with_logging', 
                        #action='store_true',
                        type=bool, 
                        default=True,
                        help='Toggle logging (saving weights and tensorboard logs)')
    parser.add_argument('--cfg_path', 
                        #action='store_true',
                        type=str, 
                        default="/home/devuser/workspace/src/configs/SemanticTHAB_default.yaml",
                        help='Path to the config file used for training')
    args = parser.parse_args()
    main(args)

    # parser.add_argument('--learning_rate', type=float, default=0.0001,
    #                     help='Learning rate for the model (default: 0.001)')
    # parser.add_argument('--num_epochs', type=int, default=50,
    #                     help='Number of epochs for training (default: 50)')
    
    # # intervals
    # parser.add_argument('--test_every_nth_epoch', type=int, default=5,
    #                     help='Test every nth epoch (default: 1)')
    # parser.add_argument('--save_every_nth_epoch', type=int, default=5,
    #                     help='Save weights every nth epoch (default: 5, when set to -1 evaluates to never but last epoch)')
    
    # parser.add_argument('--test_id', type=int, default=-1,
    #                     help='Test ID of the test sequence for the leave one out CV. -1 for training on all')
    
    # parser.add_argument('--batch_size', type=int, default=1,
    #                     help='Batch size for training (default: 8)')
    # parser.add_argument('--num_workers', type=int, default=0,
    #                     help='Number of data loading workers (default: 1, 16)')
    # parser.add_argument('--rotate', action='store_true',
    #                     help='Whether to apply rotation augmentation (default: False)')
    # parser.add_argument('--attention', action='store_true',
    #                     help='Whether to use attention (default: False)')
    # parser.add_argument('--normals', action='store_true', default=True,
    #                     help='Whether to normals as input (default: False)')
    # parser.add_argument('--multi_scale_meta', action='store_true', default=True,
    #                     help='Whether to to inject meta data at multiple scales (default: False)')
    # parser.add_argument('--pretrained', action='store_true', default=True,
    #                     help='Whether to use a model pretrained on SemanticKitti (default: False)')
    # parser.add_argument('--flip', action='store_true',
    #                     help='Whether to apply flip augmentation (default: False)')
    # parser.add_argument('--visualization', action='store_true', default=False,
    #                     help='Toggle visualization during training (default: False)')
