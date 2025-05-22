
from torch.utils.data import IterableDataset
import torch
import cv2
import numpy as np
from PIL import Image
from torch.utils.data import get_worker_info



class AptosClipDataset(IterableDataset):
    def __init__(self, dic, anno, transform):
        self.dict = dic
        self.video_annotations = anno
        self.transform = transform


    def __iter__(self):
        worker_info = get_worker_info()

        all_items = list(self.dict.items())
        if worker_info is None:
            # シングルワーカー処理
            # for single worker
            items = all_items
        else:
            # ワーカーごとに分割
            # for several workers
            total = len(all_items)
            per_worker = int(np.ceil(total / worker_info.num_workers))
            start = worker_info.id * per_worker
            end = min(start + per_worker, total)
            items = all_items[start:end]

        for case, path in items:
            video = cv2.VideoCapture(path)
            fps = video.get(cv2.CAP_PROP_FPS)

            for start, end, phase in self.video_annotations[case]:

                ts = np.linspace(start=start, stop=end, num=32, endpoint=False)
                clip_tensors = []

                for t in ts:
                    fp = int(t * fps)
                    video.set(cv2.CAP_PROP_POS_FRAMES, fp)
                    _, frame = video.read()

                    """
                    transform
                    """
                    # frame = frame / 255
                    pil_image = Image.fromarray(frame)

                    #pil_image to tensor.   range[0.0, 1.0]
                    tensor = self.transform(pil_image)
                    # tensor = tensor.to(self.device)


                    clip_tensors.append(tensor)
                clip = torch.stack(clip_tensors)
                label = torch.tensor(phase, dtype=torch.long)
                yield clip, label

            video.release()


class AptosClipDataset2(IterableDataset):
    def __init__(self, dic, anno, transform):
        self.dict = dic
        self.video_annotations = anno
        self.transform = transform


    def __iter__(self):
        worker_info = get_worker_info()

        all_items = list(self.dict.items())
        if worker_info is None:
            # シングルワーカー処理
            # for single worker
            items = all_items
        else:
            # ワーカーごとに分割
            # for several workers
            total = len(all_items)
            per_worker = int(np.ceil(total / worker_info.num_workers))
            start = worker_info.id * per_worker
            end = min(start + per_worker, total)
            items = all_items[start:end]

        for case, path in items:
            video = cv2.VideoCapture(path)
            fps = video.get(cv2.CAP_PROP_FPS)

            for start, end, phase in self.video_annotations[case]:

                ts = np.linspace(start=start, stop=end, num=32, endpoint=False)
                clip_tensors = []

                for t in ts:
                    fp = int(t * fps)
                    video.set(cv2.CAP_PROP_POS_FRAMES, fp)
                    _, frame = video.read()

                    """
                    transform
                    """
                    # frame = frame / 255
                    pil_image = Image.fromarray(frame)
                    #pil_image to tensor.     [0.0, 1.0]になっている
                    tensor = self.transform(pil_image)
                    # tensor = tensor.to(self.device)


                    clip_tensors.append(tensor)
                clip = torch.stack(clip_tensors)
                label = torch.tensor(phase, dtype=torch.long)
                yield clip, label

            video.release()













