import torch

from torch.utils.data import DataLoader
import torchvision.datasets as datasets
from datasets import load_dataset

from config_util import Config
from wds_wrapper import GenericWDS
from tokenizer.tokenizer_image.vq_model import VQModel


from dataset.build import build_dataset


def load_visual_tokenizer(config=None, ckpt=None, device="cpu"):
    # NOTE: config might be overrided by ckpt['config']
    if config is not None:
        tokenizer_config = Config(config)
    else:
        tokenizer_config = None
    
    if ckpt is not None:
        tokenizer_ckpt = torch.load(ckpt, map_location="cpu", weights_only=False)
        tokenizer_config = tokenizer_ckpt['config'] if 'config' in tokenizer_ckpt else tokenizer_config
    
    assert tokenizer_config is not None
    # create and load model
    vq_model = VQModel(tokenizer_config['model'])
    vq_model.to(device)
    vq_model.eval()

    if 'ema' in tokenizer_ckpt or 'model' in tokenizer_ckpt:
        print(vq_model.load_state_dict(tokenizer_ckpt['ema' if 'ema' in tokenizer_ckpt else 'model'], strict=False))
    else:
        print(vq_model.load_state_dict(tokenizer_ckpt, strict=False))

    vq_model = vq_model.to(device)
    vq_model.eval()
    # vq_model.requires_grad_(False)
    return vq_model
    

class CustomCollateFn:
    def __init__(self, img_key, txt_key, transform):
        self.img_key = img_key
        self.txt_key = txt_key
        self.transform = transform
    
    def __call__(self, batch):
        imgs = [self.transform(sample[self.img_key]) for sample in batch]
        if self.txt_key is not None:
            txts = [sample[self.txt_key] for sample in batch]
        else:
            txts = [torch.zeros(1) for sample in batch]
        return torch.stack(imgs), txts


def make_generic_dataset_loader(data_path, **kwargs):
    if data_path.startswith("wds://"):
        real_data_path = data_path[6:]
        dataset = GenericWDS(real_data_path, **kwargs)
        loader = dataset.make_loader()
        return dataset, loader

    if data_path.startswith("datasets://"):
        real_data_path = data_path[11:]
        print(real_data_path)
        img_key = 'jpg'
        txt_key = None
        transform = kwargs.pop("transform")
        kwargs.pop("shuffle", None)
        accelerator = kwargs.pop("accelerator", None)
        if accelerator is not None:
            with accelerator.main_process_first():
                dataset = load_dataset(real_data_path, split="train", streaming=True)
        else:
            dataset = load_dataset(real_data_path, split="train", streaming=True)
        collate_fn = CustomCollateFn(img_key, txt_key, transform)
        return dataset, DataLoader(dataset, collate_fn=collate_fn, **kwargs)

    # default
    args = kwargs.pop("args", None)
    if args is None:
        dataset = datasets.ImageFolder(data_path, transform=kwargs.pop("transform"))

    else:
        dataset = build_dataset(args, transform=kwargs.pop("transform"))
    loader = DataLoader(
            dataset,
            **kwargs)
    return dataset, loader


