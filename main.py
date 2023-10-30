from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass
import os
from datetime import datetime
from pathlib import Path
import argparse
import yaml
import wandb
import pickle

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from data_loading.data_loader import Seg2DplusT_SAX_Kspace, Seg2DplusT_SAX_Kspace_test
from model.ktst_module import KTSTModule
from log_tools.pl_callbacks import WandbLoggerCallback
from pytorch_lightning.loggers import WandbLogger

LATEST_CHECKPOINT_DIR = "latest_checkpoint"
BEST_WEIGHTS_PATH = "best_weights.pt"
CONFIG_SAVE_PATH = "config.yaml"


@dataclass
class Params:
    check_val_every_n_epoch: int = 100
    cache_data: bool = False  # Whether to load all image and segmentations to RAM
    acceleration: int = 4
    use_bboxes: bool = True
    nerf_enc_num_frequencies: Tuple[int, ...] = (4, 4, 4)
    nerf_dec_num_frequencies: Tuple[int, ...] = (4, 4, 4)
    enc_freq_scale: Tuple[float, ...] = (1.0,)
    dec_freq_scale: Tuple[float, ...] = (1.0,)
    latent_size: int = 128
    enc_embedding: str = "none"  # none, nerf, gaussian
    enc_att_token_size: int = 32
    enc_hidden_size: int = 128
    enc_num_hidden_layers: int = 2
    enc_att_num_heads: int = 2
    enc_att_max_set_size: int = 2**14
    dec_embedding: str = "none"  # none, nerf, gaussian
    dec_att_token_size: int = 32
    dec_hidden_size: int = 128
    dec_num_hidden_layers: int = 2
    dec_att_num_heads: int = 1
    dec_att_max_set_size: int = -1
    dropout: float = 0.00
    coord_noise_std: float = 1e-4
    max_epochs: int = 3001
    rec_loss_weight: float = 1.0
    seg_loss_weight: float = 1.0
    FT_rec_loss_weight: float = 1.0
    seg_class_weights: Tuple[float, ...] = (1.0, 1.0, 1.0, 1.0)
    lr: float = 1e-4
    early_stopping: bool = False
    latent_reg: float = 0.0
    enc_weight_reg: float = 1e-5
    dec_weight_reg: float = 1e-5  # 1e-5
    activation: str = "relu"  # sine, relu, wire


def get_dataset_and_model(config: Dict[str, Any], params: Params):
    train_dataset = Seg2DplusT_SAX_Kspace(load_dir=config["train_data_dir"],
                                          pickle_name=f'dataset_train_{config["num_train"]}.pkl',
                                          case_start_idx=config.get("train_start_idx", 0),
                                          num_cases=config["num_train"],
                                          **params.__dict__)
    val_dataset = Seg2DplusT_SAX_Kspace_test(load_dir=config["val_data_dir"],
                                             pickle_name=f'dataset_val_{config["num_val"]}.pkl',
                                             case_start_idx=config.get("val_start_idx"),
                                             num_cases=config["num_val"],
                                             **params.__dict__)
    test_dataset = Seg2DplusT_SAX_Kspace_test(load_dir=config["test_data_dir"],
                                              pickle_name=f'dataset_test_{config["num_test"]}.pkl',
                                              case_start_idx=config.get("test_start_idx"),
                                              num_cases=config["num_test"],
                                              **params.__dict__)
    model = KTSTModule(train_dataset=train_dataset, **params.__dict__)
    return train_dataset, val_dataset, test_dataset, model


def get_callbacks(params: Params, root_dir: str, exp_name: Optional[str] = None, is_test: Optional[bool] = False):
    callbacks = []
    # Add ckpt saver for training and define the project name
    if is_test:
        project_name = exp_name if exp_name is not None else "KINS_test"
    else:
        project_name = exp_name if exp_name is not None else "KINS_train"
        ckpt_beat_saver = ModelCheckpoint(save_top_k=1, 
                                          dirpath=root_dir,
                                          monitor="val/loss", 
                                          mode="min", 
                                          save_last=True)
        callbacks += [ckpt_beat_saver]
    # Monitor learning rate
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    callbacks += [lr_monitor]
    # Add early stopping
    if params.early_stopping:
        callbacks += [EarlyStopping(monitor="val/loss", patience=4, mode="min")]
    # Add logger
    logger = WandbLogger(project=project_name, save_dir=root_dir, config=params.__dict__)
    logger_callback = WandbLoggerCallback(project_name, log_dir=root_dir, config=params.__dict__)
    callbacks += [logger_callback]
    return callbacks, logger
    

def main_train(config_path: Optional[str] = None, exp_name: Optional[str] = None):
    # Config and hyper params
    config = {"params": {}}
    if config_path is not None:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    params = Params(**config["params"])
    root_dir = Path(config["log_dir"])

    # Dataset
    train_dataset, val_dataset, test_dataset, model = get_dataset_and_model(config, params)
    train_dataloader = DataLoader(train_dataset, shuffle=True)
    val_dataloader = DataLoader(val_dataset, shuffle=False)
    
    train_dataset[0]
    # Model dir creation
    if exp_name is not None:
        root_dir = root_dir / exp_name
    root_dir = root_dir / f'{datetime.now().strftime("%Y%m%d-%H%M%S")}_{str(train_dataset.__class__.__name__)}'
    os.makedirs(str(root_dir), exist_ok=True)
    print(root_dir)

    # Save config to model dir
    config["params"] = {k: v if not isinstance(v, tuple) else list(v) for k, v in params.__dict__.items()}
    with open(str(root_dir / CONFIG_SAVE_PATH), "w") as f:
        yaml.dump(config, f)

    # Trainer
    callbacks, logger = get_callbacks(params, root_dir, exp_name)
    trainer_kwargs = {}
    if logger is not None:
        trainer_kwargs["logger"] = logger

    trainer = pl.Trainer(max_epochs=params.max_epochs,
                         accelerator="gpu",
                         default_root_dir=root_dir,
                         callbacks=callbacks,
                         check_val_every_n_epoch=params.check_val_every_n_epoch,
                         **trainer_kwargs)

    start = datetime.now()
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    print("Train elapsed time:", datetime.now() - start)

    lastest_weights_path = root_dir / "last.ckpt"
    best_weights_path_list = [i for i in root_dir.glob("*.ckpt")
                              if i.name[:len("epoch=")] == "epoch=" and i.exists() and i.is_file()]
    best_weights_path = best_weights_path_list[0] if best_weights_path_list else None
    wandb.finish()
    return best_weights_path, params.max_epochs


def main_eval(weights_path: str, config_path: Optional[str] = None, exp_name: Optional[str] = None):
    if weights_path is None:
        raise ValueError("weights_path is required.")
    source_dir = Path(weights_path).parent

    # Load original model's config
    source_config_path = source_dir / CONFIG_SAVE_PATH
    source_config = {"params": {}}
    if source_config_path.exists():
        with open(str(source_config_path), "r") as f:
            source_config = yaml.safe_load(f)

    # Load user defined config
    config = {"params": {}}
    if config_path is not None and Path(config_path).exists():
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

    # Merged user defined config with original model config
    merged_params = {**config["params"], **source_config["params"]} # Original model config takes precedence
    params = Params(**merged_params)
    log_dir = Path(config["log_dir"]) if "log_dir" in config else Path(str(source_dir.parent))
    if exp_name is not None:
        log_dir = log_dir / exp_name
    root_dir = log_dir / (str(source_dir.name) + f'_test_{datetime.now().strftime("%Y%m%d-%H%M%S")}')
    os.makedirs(str(root_dir), exist_ok=True)

    # Define dataset and load trained model's weights
    data_config = config if "test_data_dir" in config else source_config
    _, _, test_dataset, model = get_dataset_and_model(data_config, params)
    callbacks, logger = get_callbacks(params, root_dir, exp_name, is_test=True)
    model.eval()
    sd = torch.load(weights_path)['state_dict']
    a = model.load_state_dict(sd, strict=True)
    trainer_kwargs = {}
    if logger is not None:
        trainer_kwargs["logger"] = logger
    trainer = pl.Trainer(max_epochs=1,
                        accelerator="gpu",
                        default_root_dir=root_dir,
                        enable_checkpointing=False,
                        callbacks=callbacks,
                        **trainer_kwargs)
    trainer.test(model, dataloaders=DataLoader(test_dataset, shuffle=False))

    # Save test results
    test_results = model.results
    root_dir.mkdir(exist_ok=True)
    with open(root_dir / "test_results.pkl", "wb") as f:
        pickle.dump(test_results, f)
    wandb.finish()
    return test_results


def parse_command_line():
    main_parser = argparse.ArgumentParser(description="Implicit Segmentation",
                                          formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    main_subparsers = main_parser.add_subparsers(dest='pipeline')
    # train
    parser_train = main_subparsers.add_parser("train")
    parser_train.add_argument("-c", "--config",
                              help="Path to configuration file", required=False,
                              default=r"C:\Users\nilst\Documents\Implicit_segmentation\configs\4d_cardiac_config.yaml"
                              )
    parser_train.add_argument("-n", "--exp_name",
                              help="Custom experiment name", required=False,
                              default=""
                              )
    # eval
    parser_eval = main_subparsers.add_parser("eval")
    parser_eval.add_argument("-c", "--config",
                             help="Path to configuration yml file", required=False,
                             )
    parser_eval.add_argument("-w", "--weights",
                             help="Path to the desired checkpoint .ckpt file meant for evaluation", required=True,
                             )
    parser_eval.add_argument("-n", "--exp_name",
                             help="Custom experiment name", required=False,
                             default=""
                             )
    return main_parser.parse_args()


if __name__ == '__main__':
    args = parse_command_line()
    if args.pipeline is None or args.pipeline == "train":
        config_path, exp_name = args.config, args.exp_name
        weights_path, max_epochs = main_train(config_path, exp_name)
        main_eval(weights_path, config_path, exp_name+'_test')
    elif args.pipeline == "eval":
        config_path, weights_path, exp_name = args.config, args.weights, args.exp_name
        main_eval(weights_path, config_path, exp_name)
    else:
        raise ValueError("Unknown pipeline selected.")
