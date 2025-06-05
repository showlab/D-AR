# mkdir -p /dev/shm/imagenet-val/0/
# tar -xvf ../ILSVRC2012_img_val.tar -C /dev/shm/imagenet-val/0/ >> /dev/null
import os
import random
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torchvision.utils import make_grid, save_image
from torch.utils.data import Sampler
from fvcore.nn import parameter_count_table

from dataset.augmentation import random_crop_arr
from tokenizer.tokenizer_image.vq_model import VQModel
from tokenizer.tokenizer_image.vq_train_accelerate import load_state_dict_compatible
from config_util import Config
from dar_tool import make_generic_dataset_loader

# ====================== Configurations ====================== #
CLIP_STEPS = None # jump-estimate to target from specified step, set to None for normal use
SAMPLING_STEPS = None # override for sampling steps, set to None for normal use
START_STEPS = None # only for debugging, set to None for normal use
RESOLUTION = 256
SEED = 1
DATASET = "imagenet"
DATASET_PATH = "/PATH/TO/imagenet-val/"
CKPT_PATH = "temp/tokenizer_v1.pt"
EMA = True
BATCH_SIZE = 4

imean = [0.5, 0.5, 0.5]
istd = [0.5, 0.5, 0.5]

# ====================== Seed Setup ====================== #
torch.manual_seed(SEED)

# ====================== Image Transform ====================== #
normalize = transforms.Normalize(mean=imean, std=istd)
transform = transforms.Compose([
    transforms.Lambda(lambda x: random_crop_arr(x, int(RESOLUTION))),
    transforms.ToTensor(),
    normalize,
])

# ====================== Custom Sampler ====================== #
class CustomSampler(Sampler):
    def __init__(self, indices):
        self.indices = indices
    def __iter__(self):
        return iter(self.indices)
    def __len__(self):
        return len(self.indices)

def get_indices(dataset_name):
    if dataset_name == "imagenet":
        # return [random.randint(0, 49999) for _ in range(BATCH_SIZE)]
        return [i for i in range(BATCH_SIZE)]
    return None

# ====================== Load Dataset ====================== #
indices = get_indices(DATASET)

if DATASET == "imagenet":
    val_dataset, val_loader = make_generic_dataset_loader(
        DATASET_PATH,
        transform=transform,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        sampler=CustomSampler(indices)
    )

# ====================== Utility Functions ====================== #
def decode_img_tensor(x):
    mean = x.new_tensor(imean).view(3, 1, 1)
    std = x.new_tensor(istd).view(3, 1, 1)
    return (x * std + mean).clamp(0, 1).detach().cpu()

# ====================== Load Config & Model ====================== #
config = Config("configs/tokenizer_v1.yaml")
if CKPT_PATH:
    ckpt_dict = torch.load(CKPT_PATH, map_location='cpu', weights_only=False)
    config = ckpt_dict.get("config", config)

print(f"Using model {CKPT_PATH}, resolution {RESOLUTION}, clip_steps={CLIP_STEPS}, sampling_steps={SAMPLING_STEPS}, start_steps={START_STEPS}")

if SAMPLING_STEPS is not None:
    config["model"]["num_denoising_steps"] = SAMPLING_STEPS
config["model"]["sampling_resolution"] = RESOLUTION

model = VQModel(config["model"]).cuda().eval()
model.requires_grad_(False)
print(parameter_count_table(model, max_depth=1))

if CKPT_PATH:
    if "ema" in ckpt_dict or "model" in ckpt_dict:
        weights = ckpt_dict["ema" if EMA else "model"]
    else:
        weights = ckpt_dict
    skipped = load_state_dict_compatible(model, weights)['skipped']
    print("Skipped weights:", skipped)

# ====================== Inference & Visualization ====================== #
for x, y in val_loader:
    break  # Only one batch

with torch.inference_mode(), torch.autocast("cuda", enabled=True, dtype=torch.bfloat16):
    x = x.cuda()
    if config['model'].get('vq_enable', True):
        code = model.extract_code(x)
        print("Dead rate:", (code.float().std(0) < 1).float().mean())
        print("Sample Code (first sample):", code[0])
        torch.manual_seed(SEED)
        decoded_imgs = model.decode_code(code, height=RESOLUTION, width=RESOLUTION, clip_steps=CLIP_STEPS, start_steps=START_STEPS, start_x=x).cpu()
    else:
        decoded_imgs = model.straight_forward(x, clip_steps=CLIP_STEPS).cpu()

print("Labels:")
print("\n".join([str(label) for label in y]))

# ====================== Save Visualization ====================== #
grid = make_grid(
    [decode_img_tensor(x[i]) for i in range(BATCH_SIZE)] +
    [decode_img_tensor(decoded_imgs[i]) for i in range(BATCH_SIZE)],
    nrow=BATCH_SIZE,
    normalize=True
)
save_image(grid, "vq.png")
print("Image saved to vq.png")
