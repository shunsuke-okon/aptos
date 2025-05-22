import yaml
from pathlib import Path
import os
from torch.utils.data import DataLoader
from torchvision import transforms
from torch import optim
import torch
import torch.nn as nn


# オリジナル
# original
from tools import get_train_val
from custom_data import AptosClipDataset
from model import Model


def sample():
    # yamlファイルからセッティングデータを取得
    # get setting data from yaml file
    with open("config.yaml", "r") as f:
        conf = yaml.safe_load(f)

    # pathをyamlファイルが扱える形に変更する
    # change path form
    conf["data_path"] = conf["data_path"].removeprefix("r\"").strip("\"")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_path = Path(conf["data_path"])

    # print(os.listdir(videos_path))

    train, val, video_annotations = get_train_val(data_path)

    tran_list = [transforms.Resize((conf["image_size"], conf["image_size"])),
                 transforms.ToTensor(), ]
    transform_train = transforms.Compose(tran_list)

    ds = AptosClipDataset(train, video_annotations, transform_train)

    train_loader = DataLoader(
        ds,
        batch_size=conf["batch_size"],
        num_workers=conf["num_workers"],
    )
    it = iter(train_loader)

    














if __name__ == "__main__":
    sample()