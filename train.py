import yaml
from pathlib import Path
import os
from torch.utils.data import DataLoader
from torchvision import transforms
from torch import optim
import torch
import torch.nn as nn
from datetime import datetime



# オリジナル
# original
from tools import get_train_val
from custom_data import AptosClipDataset
from model import Model


def train():
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
    # print(train)
    # print(val)
    # print(video_annotations)

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

    """
    prepare
    """
    model = Model().to(device=device)
    model.train()
    epoch = conf["epoch"]
    optimizer = optim.AdamW(model.parameters(),
                            lr=conf["lr"],
                            weight_decay=conf["weight_decay"])
    ce_loss = nn.CrossEntropyLoss()

    # 現在の日付と時刻を取得
    # get date
    now = datetime.now()
    # フォーマットした日付と時間（時、分、秒）を取得
    # get formatted date
    formatted_date_time = now.strftime('%Y-%m-%d_%H-%M-%S')

    model_save_path = Path(__file__).parent / "model_param" / formatted_date_time
    model_save_path.mkdir(parents=True, exist_ok=True)


    """
    training
    """
    for epo in range(epoch):
        clips, label = next(it)

        # optimizer.zero_grad()
        # output = model(clips)
        # loss = ce_loss(output, )
        # loss.backward()
        # optimizer.step()

        if epo % conf["save_interval"]:
            file_name = f"model_{epo}_epoch.pt"
            torch.save(model.state_dict(), os.path.join(model_save_path, file_name))













if __name__ == "__main__":
    train()
