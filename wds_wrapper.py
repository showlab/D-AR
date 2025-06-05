import os
import math
import time
import random
import warnings
from glob import glob
from copy import deepcopy

import webdataset as wds

from torch.utils.data import default_collate
from huggingface_hub import HfFileSystem, get_token, hf_hub_url




class ImageNetWDS:
    def __init__(self, tar_dir, transform, batch_size=64, num_workers=4, global_batch_size=256):
        self.tar_dir = tar_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transform
        self.global_batch_size = global_batch_size
        # self.num_total_samples = 1_300_000
        self.num_total_samples = 1281167

    def make_loader(self):
        shards = os.listdir(self.tar_dir)
        shards = sorted([os.path.join(self.tar_dir, s) for s in shards if s.endswith(".tar") and "train" in s])

        train_processing_pipeline = [
                wds.decode(wds.autodecode.ImageHandler("pil", extensions=["webp", "png", "jpg", "jpeg"])),
                wds.to_tuple("jpg;png", "cls"),
                wds.map_tuple(
                    self.transform, int
                ),
            ]

        pipeline = [
            wds.ResampledShards(shards),
            wds.tarfile_to_samples(handler=wds.warn_and_continue),
            wds.shuffle(bufsize=5000,
                        initial=1000),
            *train_processing_pipeline,
            wds.batched(self.batch_size, partial=False, collation_fn=default_collate),
        ]

        num_worker_batches = math.ceil(self.num_total_samples / (self.global_batch_size * self.num_workers))

        dataset = wds.DataPipeline(*pipeline).with_epoch(num_worker_batches)
        

        dataloader = wds.WebLoader(
            dataset,
            num_workers=self.num_workers,
            batch_size=None, pin_memory=True, persistent_workers=True
        )
        
        return dataloader
    

class GenericWDS:
    def __init__(self, tar_dir, transform, batch_size=64, num_workers=4, global_batch_size=256, target_type="c2i", num_total_samples=0):
        self.tar_dir = tar_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transform
        self.global_batch_size = global_batch_size
        self.num_total_samples = num_total_samples
        self.target_type = target_type

    def make_loader(self):
        if self.tar_dir.startswith("hf://"):
            fs = HfFileSystem()
            files = [fs.resolve_path(path) for path in fs.glob(self.tar_dir)]
            shards = [hf_hub_url(file.repo_id, file.path_in_repo, repo_type="dataset") for file in files]
            shards = f"pipe: curl -s -L -H 'Authorization:Bearer {get_token()}' {'::'.join(shards)}"
        else:
            shards = os.listdir(self.tar_dir)
            shards = sorted([os.path.join(self.tar_dir, s) for s in shards if s.endswith(".tar")])

        if self.target_type == "c2i":
            text_transform = str
        else:
            text_transform = lambda x: 0

        train_processing_pipeline = [
                wds.decode(wds.autodecode.ImageHandler("pil", extensions=["webp", "png", "jpg", "jpeg"])),
                wds.to_tuple("jpg", "txt"),
                wds.map_tuple(
                    self.transform, text_transform
                ),
            ]

        pipeline = [
            wds.ResampledShards(shards),
            wds.tarfile_to_samples(handler=wds.warn_and_continue),
            wds.shuffle(bufsize=2000,
                        initial=1000),
            *train_processing_pipeline,
            wds.batched(self.batch_size, partial=False, collation_fn=default_collate),
        ]

        if self.num_total_samples > 0:
            num_worker_batches = math.ceil(self.num_total_samples / (self.global_batch_size * self.num_workers))
            dataset = wds.DataPipeline(*pipeline).with_epoch(num_worker_batches)
        else:
            dataset = wds.DataPipeline(*pipeline)
        

        dataloader = wds.WebLoader(
            dataset,
            num_workers=self.num_workers,
            batch_size=None, pin_memory=True, persistent_workers=True
        )
        
        return dataloader