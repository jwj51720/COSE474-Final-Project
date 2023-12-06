import argparse
from experiments import *
import os
import torch


def main(config):
    model = load_model(config)
    model.load_state_dict(torch.load(f"{config['save_path']}/current_best.pt"))
    testloader = test_dataloader(config)
    inference(config, model, testloader)
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
