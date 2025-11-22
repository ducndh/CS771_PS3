import os
import random

import numpy as np
import copy

import torch
import torchvision
from torch.utils.data import DataLoader
import torchvision.transforms as T

from .transforms import Compose, ConvertAnnotations, RandomHorizontalFlip, ToTensor
from torch.utils.data.distributed import DistributedSampler


def trivial_batch_collator(batch):
    """
    A batch collator that allows us to bypass auto batching
    """
    return tuple(zip(*batch))


def worker_init_reset_seed(worker_id):
    """
    Reset random seed for each worker
    """
    seed = torch.initial_seed() % 2**31
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


class VOCDetection(torchvision.datasets.CocoDetection):
    """
    A simple dataset wrapper to load VOC data
    """

    def __init__(self, img_folder, ann_file, transforms):
        super().__init__(img_folder, ann_file)
        self._transforms = transforms

    def get_cls_names(self):
        cls_names = (
            "aeroplane",
            "bicycle",
            "bird",
            "boat",
            "bottle",
            "bus",
            "car",
            "cat",
            "chair",
            "cow",
            "diningtable",
            "dog",
            "horse",
            "motorbike",
            "person",
            "pottedplant",
            "sheep",
            "sofa",
            "train",
            "tvmonitor",
        )
        return cls_names

    def __getitem__(self, idx):
        img, target = super().__getitem__(idx)
        image_id = self.ids[idx]
        target = dict(image_id=image_id, annotations=target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target


class COCODetection(torchvision.datasets.CocoDetection):
    """
    A simple dataset wrapper to load COCO data
    """

    def __init__(self, img_folder, ann_file, transforms):
        super().__init__(img_folder, ann_file)
        self._transforms = transforms
        self.label_ids = sorted(self.coco.getCatIds())
        self.id_map, self.rev_id_map = self._convert_to_contiguous_ids()

    def get_cls_names(self):
        cats = self.coco.loadCats(self.label_ids)
        nms=[cat['name'] for cat in cats]
        return nms
    
    def _convert_to_contiguous_ids(self):
        cnt = 1
        d = {}
        d_rev = {}
        for id in self.label_ids:
            d[id] = cnt
            d_rev [cnt] = id
            cnt += 1
        return d, d_rev

    def __getitem__(self, idx):
        img, t = super().__getitem__(idx)
        target = copy.deepcopy(t)
        image_id = self.ids[idx]
        for obj in target:
            obj["category_id"] = self.id_map[obj["category_id"]]
        target = dict(image_id=image_id, annotations=target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target


def build_dataset(name, split, img_folder, json_folder):
    """
    Create VOC/COCO dataset with default transforms for training / inference.
    New datasets can be linked here.
    """
    if name == "VOC2007":
        assert split in ["trainval", "test"]
        is_training = split == "trainval"
    elif name == "COCO":
        assert split in ["train", "val"]
        is_training = split == "train"
    else:
        print("Unsupported dataset")
        return None

    if is_training:
        transforms = Compose([ConvertAnnotations(), RandomHorizontalFlip(), ToTensor()])
    else:
        transforms = Compose([ConvertAnnotations(), ToTensor()])

    if name == "VOC2007":
        dataset = VOCDetection(
            img_folder, os.path.join(json_folder, split + ".json"), transforms
        )
    elif name == "COCO":
        ann_file = os.path.join(json_folder, f"instances_{split}2017.json")
        # img_folder_split = os.path.join(img_folder, f"{split}2017")
        dataset = COCODetection(img_folder, ann_file, transforms)

    return dataset


def build_dataloader(dataset, is_training, batch_size, num_workers):
    """
    Create a dataloder for VOC dataset
    """
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=trivial_batch_collator,
        worker_init_fn=(worker_init_reset_seed if is_training else None),
        shuffle=is_training,
        drop_last=is_training,
        persistent_workers=True,
        # sampler = DistributedSampler(dataset),
    )
    return loader
