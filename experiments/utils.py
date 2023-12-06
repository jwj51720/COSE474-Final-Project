import sys

sys.path.append("../")

import torch
import numpy as np
import json
from pathlib import Path
from collections import OrderedDict
from layers import ResNet, BasicBlock, BottleNeck, NewModel
import torchvision.models as models
import torchvision.transforms as transforms


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)


def read_json(fname):
    fname = Path(fname)
    with fname.open("rt") as handle:
        return json.load(handle, object_hook=OrderedDict)


def load_model(config):
    if config["new_model_flag"]:
        config["classes1"] = config["classes"][: int(len(config["classes"]) / 2)]
        config["classes2"] = config["classes"][int(len(config["classes"]) / 2) :]
        print(config["classes1"])
        print(config["classes2"])
        num_classes = len(config["classes1"]) + 1
    else:
        num_classes = len(config["classes"])

    if config["model"] == "resnet18":
        model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes)
    elif config["model"] == "resnet34":
        model = ResNet(BasicBlock, [3, 4, 6, 3], num_classes)
    elif config["model"] == "resnet50":
        model = ResNet(BottleNeck, [3, 4, 6, 3], num_classes)
    elif config["model"] == "resnet101":
        model = ResNet(BottleNeck, [3, 4, 23, 3], num_classes)
    elif config["model"] == "resnet152":
        model = ResNet(BottleNeck, [3, 8, 36, 3], num_classes)
    else:
        model = models.resnet18(pretrained=True)

    if config["new_model_flag"]:
        print("new model!")
        model = NewModel(config, model)
    return model.to(config["device"])


def img_transform():
    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    return transform
