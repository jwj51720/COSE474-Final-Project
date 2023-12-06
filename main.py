import argparse
from experiments import *
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter


def main(config):
    model = load_model(config)
    writer = SummaryWriter(
        log_dir=f'{config["save_path"]}/{config["tensorboard_path"]}/{config["model"]}_New_{config["new_model_flag"]}'
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"] / 2)
    fold_loaders = train_dataloader(config)

    trainer(config, fold_loaders[0], model, writer, criterion, optimizer, scheduler)
    return 0


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="COSE474")
    args.add_argument(
        "-c",
        "--config",
        default="./experimental_setting.json",
        type=str,
    )
    args = args.parse_args()
    config = read_json(args.config)
    config["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    if not os.path.exists(config["save_path"]):  # default: ./outputs
        os.makedirs(config["save_path"])
    set_seed(config["seed"])
    main(config)
