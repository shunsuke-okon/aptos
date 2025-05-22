
from collections import defaultdict

import pandas as pd
import numpy as np
from datasets import load_dataset, Features, Value, Video
from huggingface_hub import hf_hub_download
import torch
from torchvision import transforms
from torch.utils.data import IterableDataset, DataLoader
from PIL import Image
import cv2
from torch.utils.data import get_worker_info


# video_path = r"F:\DATASET\Phase_Recognition_of_Surgical_Videos_Using_ML\aptos_videos\case_0012.mp4"
# img_path = r"F:\DATASET\FAZ_segmentation\superficial\original\1.jpg"

class Iter(IterableDataset):
    def __init__(self, list_a):
        self.list = list_a

    def __iter__(self):
        worker_info = get_worker_info()

        all_items = list(self.list)
        if worker_info is None:
            # シングルワーカー処理
            items = all_items
        else:
            # ワーカーごとに分割
            total = len(all_items)
            per_worker = int(np.ceil(total / worker_info.num_workers))
            start = worker_info.id * per_worker
            end = min(start + per_worker, total)
            items = all_items[start:end]
            print(f"worker_id:{worker_info.id},{items}")

            for item in items:
                yield item







if __name__ == "__main__":
    a = [1,2,3,4,5,56,7,43,45]

    ds = Iter(a)
    it = DataLoader(
        ds,
        batch_size=4,
        num_workers=2
    )
    it = iter(it)
    nt = next(it)
    print(nt)
