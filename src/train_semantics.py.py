import glob
import argparse
import torch
from torch.utils.data import DataLoader
#from models.semanticFCN import SemanticNetworkWithFPN
import torch.optim as optim

import os
import sys

from dataset.definitions import color_map, class_names
from utils.weights import load_pretrained_safely

import json
import yaml

import time

import matplotlib
matplotlib.use("TkAgg")  # often more robust than Tk

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
    assert args.mode in {"train", "test"}, "Set script mode at argumentparser to be one of 'train' OR 'test'"
    cfg = load_config(args.cfg_path)

    # add additional configurations to config file
    cfg["extras"] = dict()
    cfg["extras"]["use_reflectivity"] = True
    cfg["extras"]["num_classes"] = 21 if (cfg['dataset_name']=="SemanticSTF" or cfg['dataset_name']=="SemanticWADS") \
                                    and not cfg.get('remap_adverse_label', 0)  else 20
    cfg["extras"]["class_names"] = class_names
    cfg["extras"]["class_colors"] = color_map
    #cfg["extras"]["loss_function"] = "Tversky"    # Tversky | Dirichlet
    # choose model architecture, type str
    baseline = cfg["model_settings"].get("baseline", "Reichert")
    cfg["model_settings"]["baseline"] = baseline    # ensure that the basline parameter is set if not pre-initialized in config
    #num_folder = count_folders(cfg["dataset_dir"])

    match cfg['dataset_name']:
        case "SemanticSTF":
            data_path_train = [(bin_path, bin_path.replace("velodyne", "labels").replace("bin", "label")) for bin_path in glob.glob(f"{cfg['dataset_dir']}/train/velodyne/*.bin")]
            data_path_test = [(bin_path, bin_path.replace("velodyne", "labels").replace("bin", "label")) for bin_path in glob.glob(f"{cfg['dataset_dir']}/val/velodyne/*.bin")]
        case "Panoptic-CUDAL":
            data_path_train = [(bin_path, bin_path.replace("velodyne", "labels").replace("bin", "label")) for folder in [f"{i:02}" for i in [30, 31, 32, 36, 40, 41]] for bin_path in glob.glob(f"{cfg['dataset_dir']}/{folder}/velodyne/*.bin")]
            data_path_test = [(bin_path, bin_path.replace("velodyne", "labels").replace("bin", "label")) for bin_path in glob.glob(f"{cfg['dataset_dir']}/34/velodyne/*.bin")]  # use sequence 8 as validation set
        case "SemanticTHAB":
            data_path_train = [(bin_path, bin_path.replace("velodyne", "labels").replace("bin", "label")) for folder in [f"{i:04}" for i in range(9) if i != 6] for bin_path in glob.glob(f"{cfg['dataset_dir']}/{folder}/velodyne/*.bin")]
            data_path_test = [(bin_path, bin_path.replace("velodyne", "labels").replace("bin", "label")) for bin_path in glob.glob(f"{cfg['dataset_dir']}/0006/velodyne/*.bin")] 
        case "SemanticWADS":    
            data_path_train = [(bin_path, bin_path.replace("velodyne", "labels").replace("bin", "label")) for folder in [f"{i:02}" for i in [11, 12, 13, 14, 15, 16, 17, 18, 20, 22, 23, 24, 26, 28, 34, 35, 36, 37, 76]] for bin_path in glob.glob(f"{cfg['dataset_dir']}/{folder}/velodyne/*.bin")]
            data_path_test = [(bin_path, bin_path.replace("velodyne", "labels").replace("bin", "label")) for bin_path in glob.glob(f"{cfg['dataset_dir']}/30/velodyne/*.bin")] 
        
        case _: 
            try:
                data_path_train = [(bin_path, bin_path.replace("velodyne", "labels").replace("bin", "label")) for folder in [f"{i:02}" for i in range(11) if i != 8] for bin_path in glob.glob(f"{cfg['dataset_dir']}/{folder}/velodyne/*.bin")]
                data_path_test = [(bin_path, bin_path.replace("velodyne", "labels").replace("bin", "label")) for bin_path in glob.glob(f"{cfg['dataset_dir']}/08/velodyne/*.bin")]  # use sequence 8 as validation set
            except Exception as ex:
                print(ex)
    # Dataloader import for datasets derived from SemanticKitti
    match cfg['dataset_name']:
        case "SemanticKitti": from dataset.dataloader_semantic_KITTI import SemanticKitti as SemanticDataset
        case "SemanticSTF": from dataset.dataloader_semantic_STF import SemanticSTF as SemanticDataset
        case "SemanticTHAB": from dataset.dataloader_semantic_THAB import SemanticTHAB as SemanticDataset
        case "Panoptic-CUDAL": from dataset.dataloader_semantic_CUDAL import SemanticCUDAL as SemanticDataset
        case "SemanticWADS": from dataset.dataloader_semantic_WADS import SemanticWADS as SemanticDataset
        case _: raise KeyError("in yaml config parameter dataset_name is invalid")
    
    depth_dataset_train = SemanticDataset(
                            data_path=data_path_train, 
                            rotate=cfg["model_settings"]["rotate"],
                            flip=cfg["model_settings"]["flip"],
                            projection=cfg['model_settings'].get('projection', [64,512]),
                            resize=cfg['model_settings'].get('resize', False))
    # Adjustment for SemanticSTF dataset: 
        # SemanticKitti has a default of 20 classes, SemanticSTF adds one adverse weather/corruption class labeled as "20: 'invalid'"
        # Choose in SemanticSTF config yaml whether to train on 21 classes or remap the adverse weather class to "0: 'unlabeled'"
    if (cfg['dataset_name']=="SemanticSTF" or cfg['dataset_name']=="SemanticWADS") \
        and cfg.get('remap_adverse_label', 0): depth_dataset_train.remap_adverse_label = True
    
    dataloader_train = DataLoader(
                            dataset=depth_dataset_train,
                            batch_size=cfg["train_params"]["batch_size"],
                            shuffle=True,
                            num_workers=cfg["train_params"]["num_workers"],
                            pin_memory=True,                                     # required for non_blocking H2D
                            persistent_workers=True,
                            prefetch_factor=4
                        )
    
    depth_dataset_test = SemanticDataset(
                            data_path=data_path_test,
                            rotate=False,
                            flip=False,
                            projection=cfg['model_settings'].get('projection', [64,512]),
                            resize=cfg['model_settings'].get('resize', False))   # TODO: Change selection, currently reduced
    dataloader_test = DataLoader(depth_dataset_test, batch_size=1, shuffle=False, num_workers=cfg["train_params"]["num_workers"])
    
    # defines model architecture and input dimensions
    # generally: input channels minimium of range image (1ch) and xyz image (3ch)
    if cfg["model_settings"]["baseline"]=="Reichert":
        # import model
        #from models.semantic_dirichletFCN import SemanticNetworkWithFPN
        from baselines.Reichert.semanticFCN_opt import SemanticNetworkWithFPN
        # here: input channels are split into two paths "input_channels" and "meta_channel_dim"
        input_channels = 1      # min. range image (=1ch)
        meta_channel_dim = 3    # min. xyz image (=3ch)
        # reflectivity adds 1 extra channel to input_channels
        if cfg["model_settings"].get("reflectivity", 0):    input_channels +=1
        # normals adds 3 extra channels to meta_channel_dim
        if cfg["model_settings"].get("normals", 0):         meta_channel_dim +=3  

        # Model definition for "Reichert"
        model = SemanticNetworkWithFPN(
            backbone=cfg["model_settings"]["model_type"],
            input_channels=input_channels,
            meta_channel_dim=meta_channel_dim,
            num_classes=cfg["extras"]["num_classes"],
            attention=cfg["model_settings"]["attention"],
            multi_scale_meta=cfg["model_settings"]["multi_scale_meta"]
        )
    elif cfg["model_settings"]["baseline"] in {"SalsaNext", "SalsaNextAdf"}:
        # import base SalsaNext or uncertainty yielding model SalsaNextAdf
        if cfg["model_settings"]["baseline"]=="SalsaNext":      from baselines.SalsaNext.SalsaNext import SalsaNext
        elif cfg["model_settings"]["baseline"]=="SalsaNextAdf": from baselines.SalsaNext.SalsaNextAdf import SalsaNextUncertainty as SalsaNext
        
        # minimum channels=4: range image (1ch) and xyz image (3ch)
        nchannels = 4
        # normals adds 3 extra channels
        if cfg["model_settings"].get("normals", 0):         nchannels +=3
        # reflectivity adds 1 extra channel
        if cfg["model_settings"].get("reflectivity", 0):    nchannels +=1  
        
        # Model definition
        model = SalsaNext(nclasses=cfg["extras"]["num_classes"], nchannels=nchannels)
    
    # remove activation function from last layer of decoder
    # if cfg["extras"]["loss_function"] == "Dirichlet":
    #     del model.decoder_semantic[-1]
    # count number of model parameters
    num_params = count_parameters(model)
    print("num_params", num_params)
    cfg["extras"]["num_params"] = num_params
    
    ###--- Safe-load model weights ---###
    #####################################
    rep = load_pretrained_safely(
        model,
        cfg['model_settings']['pretrained'],
        device="cuda",
        ignore_keys_with=("logits.",),     # skip classification head (mismatch 20 vs 21)
        copy_head_overlap=False,           # set True if you want partial row copy for heads
        verbose=True,
    )
    if not rep.get("ok"):
        print("No pretrained weights found or applied. Training from scratch...")
        time.sleep(1)
    #####################################
    
    # Define optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg["train_params"].get("learning_rate", 1e-3),
        weight_decay=cfg["train_params"].get("weight_decay", 1e-4)  # typical: 1e-2 -> 1e-4
    )
    # Linear warm-up over first N epochs
    num_warmup_epochs = cfg["train_params"].get("num_warmup_epochs", 2)
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.3,   # start at 30% of base LR, or higher when using small batches (e.g., <16)
        end_factor=1.0,     # ramp to 100% of base LR
        total_iters=num_warmup_epochs
    )
    #Cosine annealing with restarts thereafter
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=15,           # first restart after T_0 epochs, e.g., 15
        T_mult=1,         # no increase in cycle length
        eta_min=5e-6      # floor LR
    )  
    #Chain them: warm-up then cosine restarts
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[num_warmup_epochs]
    ) 
    
    # ReduceLROnPlateau
    #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = "min", factor = 0.1)
    
    if args.with_logging:
        # Save Path
        if cfg["logging_settings"]["test_id"] != -1:
            save_path_p1 =f"{cfg['logging_settings']['log_dir']}/{cfg['model_settings']['baseline']}/test_split_{str(cfg['logging_settings']['test_id']).zfill(4)}"
        else:
            save_path_p1 =f"{cfg['logging_settings']['log_dir']}/{cfg['model_settings']['baseline']}/test_split_final"
        
        save_path_p2 ='{}_{}{}{}{}{}'.format(cfg["model_settings"]["model_type"], "a" if cfg["model_settings"]["attention"] else "", "n" if cfg["model_settings"]["normals"] else "", "m" if cfg["model_settings"]["multi_scale_meta"] else "", "p" if cfg["model_settings"]["pretrained"] else "", cfg["model_settings"]["loss_function"])
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
    if args.mode=="train":
        from models.trainer import Trainer
        trainer = Trainer(model, optimizer, cfg, scheduler = scheduler, visualize = args.visualization, logging=args.with_logging)
        trainer(dataloader_train, dataloader_test)
    elif args.mode=="test":
        from models.tester import Tester
        tester = Tester(
            model=model,
            cfg=cfg,
            visualize=args.visualization,
            logging=args.with_logging,
            checkpoint=cfg["model_settings"].get("pretrained")  # or a separate path
        )

        do_calib = cfg["calibration"].get("enable", False)
        tester.run(
            dataloader_test=dataloader_test,
            calib_loader=dataloader_test,   # or a dedicated calib loader
            do_calibration=do_calib,
            ts_mode="mc" if cfg['model_settings']['use_dropout'] else "default",              # or "mc" (averages logits with dropout)
            mc_samples=30
        )
    
    #cfg["train_params"]["num_epochs"], test_every_nth_epoch=cfg["logging_settings"]["test_every_nth_epoch"], save_every_nth_epoch=cfg["logging_settings"]["save_every_nth_epoch"])

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
                        default="/home/devuser/workspace/src/configs/SemanticKitti_default.yaml",
                        help='Path to the config file used for training')
    parser.add_argument('--mode', 
                        #action='store_true',
                        type=str, 
                        default="train",
                        help="Training option whether to 'train' or 'test' the model")
    args = parser.parse_args()
    main(args)
