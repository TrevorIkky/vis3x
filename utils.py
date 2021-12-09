import argparse
import os
import random
import sys

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from PIL import ImageFilter, ImageOps


class GaussianBlur(object):
    def __init__(self, p=0.5, min_radius=0.1, max_radius=0.2):
        self.p = p
        self.min_radius = min_radius
        self.max_radius = max_radius

    def __call__(self, img):
        blur = random.random() <= self.p
        if not blur:
            return img
        return img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.min_radius, self.max_radius)
            )
        )


class Solarize(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        solarize = random.random() < self.p
        if not solarize:
            return img
        return ImageOps.solarize(img)


class MultiCropWrapper(nn.Module):
    def __init__(self, backbone, head):
        super(MultiCropWrapper, self).__init__()
        self.head = head
        self.backbone = backbone
        backbone.head = nn.Identity()

    def forward(self, x):
        if not isinstance(x, list):
            x = [x]
        L = torch.tensor([img.shape[-1] for img in x])
        crop_idxs = torch.cumsum(torch.unique_consecutive(L, return_counts=True)[1], 0)
        start_idx, output = 0, torch.empty(0, device=x[0].device)

        for end_idx in crop_idxs:
            _out = self.backbone(torch.cat(x[start_idx:end_idx]))
            if isinstance(_out, tuple):
                _out = _out[0]
            output = torch.cat([output, _out])
            start_idx = end_idx

        # Pass the concatenated features to the head
        return self.head(output)


def bool_flag(s):
    """
    Parse boolean arguments from the command line.
    """
    FALSY_STRINGS = {"off", "false", "0"}
    TRUTHY_STRINGS = {"on", "true", "1"}
    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError("invalid value for a boolean flag")


# Copied from DINO repository

def init_distributed_mode(args):
    # launched with torch.distributed.launch
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    # launched with submitit on a slurm cluster
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    # launched naively with `python main_dino.py`
    # we manually add MASTER_ADDR and MASTER_PORT to env variables
    elif torch.cuda.is_available():
        print('Will run the code on one GPU.')
        args.rank, args.gpu, args.world_size = 0, 0, 1
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29500'
    else:
        print('Does not support training without GPU.')
        sys.exit(1)

    dist.init_process_group(
        backend="nccl",
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
    )

    torch.cuda.set_device(args.gpu)
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    dist.barrier()
    setup_for_distributed(args.rank == 0)


# Copied From DINO repository

def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_available_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_available_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_available_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def create_ckpt_dir(path):
    if is_main_process() and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def get_mean_and_std(data_loader):
    # var(x) = E(x**2) - E(x)**2
    channels_sum, channels_sum_sq, n_batches = 0, 0, 0

    for img in data_loader:
        channels_sum += torch.mean(img, dim=[0, 2, 3])
        channels_sum_sq += torch.mean(img ** 2, dim=[0, 2, 3])
        n_batches += 1

    mean = channels_sum / n_batches
    std = (channels_sum / n_batches - mean) ** 0.5

    return mean, std


def has_batch_norm(model):
    batch_norms = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)
    for name, module in model.named_modules():
        if isinstance(module, batch_norms):
            return True
    return False


def cosine_scheduler(base_val, final_val, epochs, n_iters_per_epoch, warmup_epochs=0, warmup_val=0):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * n_iters_per_epoch
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(warmup_val, base_val, warmup_iters)

    iters = np.arange((epochs - warmup_epochs) * n_iters_per_epoch)
    schedule = final_val + 0.5 * (base_val - final_val) * (1 + np.cos(np.pi * iters / (len(iters))))
    schedule = np.concatenate([warmup_schedule, schedule])

    assert len(schedule) == epochs * n_iters_per_epoch
    return schedule


def restart_from_ckpt(ckpt_path=None, restore_point=None, **kwargs):
    if not os.path.isfile(ckpt_path):
        print(f"Checkpoint: {ckpt_path} is not a file.")
        return
    print(f"Loading ckpt from: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    for key, val in kwargs.items():
        if key in checkpoint and val is not None:
            try:
                message = val.load_state_dict(checkpoint['key'], strict=False)
                print(f"==> Loaded {key} from ckpt {ckpt_path} with {message}")
            except TypeError:
                try:
                    message = val.load_state_dict(checkpoint['key'])
                    print(f"==> Loaded {key} from ckpt {ckpt_path} with {message}")
                except ValueError:
                    print(f"==> Unable to load {key} from ckpt {ckpt_path}")
        else:
            print(f"==> Unable to load {key} because {key} not in ckpt")

    # Set starting epoch
    if restore_point is not None:
        for key in restore_point:
            if key in checkpoint:
                restore_point[key] = checkpoint[key]


def get_param_groups(model):
    regularized = []
    not_regularized = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.endswith(".bias") or len(param.shape) == 1:
            not_regularized.append(param)
        else:
            regularized.append(param)
    return [{"params": regularized}, {"params": not_regularized, "weight_decay": 0.}]


def clip_gradients(model, clip):
    norms = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2)
            norms.append(param_norm)
            clip_coef = clip / (param_norm + 1e-6)
            if clip_coef < 1:
                param.grad.mul_(clip)
    return norms


def cancel_gradients_last_layer(epochs, model, freeze_last_layer):
    if epochs >= freeze_last_layer:
        return
    for name, param in model.named_parameters():
        if "last_layer" in name:
            param.grad = None
