import glob
import argparse
import torch
from torch.utils.data import DataLoader
from dataset.dataloader_semantic_THAB import SemanticKitti
from models.semanticFCN import SemanticNetworkWithFPN

import torch.optim as optim

import os

from dataset.definitions import color_map, class_names

from models.tester import Tester
from models.trainer import Trainer
import json


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main(args):
    with open(args.save_path) as json_data:
        config = json.load(json_data)
        json_data.close()
    
    data_path_test = [(bin_path, bin_path.replace("velodyne", "labels").replace("bin", "label")) for bin_path in glob.glob(f"/home/appuser/data/SemanticTHAB/sequences/0006/velodyne/*.bin")]
    depth_dataset_test = SemanticKitti(data_path_test, rotate=False, flip=False)
    dataloader_test = DataLoader(depth_dataset_test, batch_size=1, shuffle=False, num_workers=8)
    
    # Semantic Segmentation Network
    if config["USE_NORMALS"]:
        model = SemanticNetworkWithFPN(backbone=config["BACKBONE"], meta_channel_dim=6, num_classes=20, attention=config["USE_ATTENTION"], multi_scale_meta=config["USE_MULTI_SCALE"])
    else:
        model = SemanticNetworkWithFPN(backbone=config["BACKBONE"], meta_channel_dim=3, num_classes=20, attention=config["USE_ATTENTION"], multi_scale_meta=config["USE_MULTI_SCALE"])

    num_params = count_parameters(model)

    save_path = config["SAVE_PATH"]
    # test final model
    tester = Tester(model, save_path=os.path.join(save_path, "model_final.pth"), config=config, load=True)
    tester(dataloader_test)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Test Script For Semantic Kitti')

    parser.add_argument('--save_path', type=str, default='/home/appuser/data/train_semantic_THAB_v2/ablation/test_split_0006/resnet34_amp/config.json',
                        help='path to config.json')
    args = parser.parse_args()

    main(args)
