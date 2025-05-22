import csv
import os
from collections import defaultdict

def get_train_val(data_folder_path):
    video_folder = os.path.join(data_folder_path, "aptos_videos")
    train_val_csv_path = os.path.join(data_folder_path, "APTOS_train-val_annotation.csv")
    videos_list = sorted(os.listdir(video_folder))

    train = {}
    val = {}
    video_annotations = defaultdict(list)

    train_or_val = {}  # dict
    with open(train_val_csv_path) as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            train_or_val[row[0]] = row[3]
            video_annotations[row[0]].append((float(row[1]), float(row[2]), int(row[4])))
            # print(row)
        # print(train_or_val)

    for file in videos_list:
        case = file.split(".")[0]
        if train_or_val[case] == "train":
            file = os.path.join(video_folder, file)
            train[case] = file
        else:
            file = os.path.join(video_folder, file)
            val[case] = file
    return train, val, video_annotations
