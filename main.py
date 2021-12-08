import argparse
import math
import os.path
import sys

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import transforms as T
from tqdm import tqdm
from torchvision.transforms import InterpolationMode
import utils
import vision_transformers as vit
from dataset import Vis3xDataset
from vision_transformers import DINOHead


class DataAugmentationsDINO(object):
    def __init__(self, global_image_scale, local_image_scale, num_local_crops):
        flip_and_color_jitter = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply([T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)], p=0.8),
            T.RandomGrayscale(p=0.2)
        ])
        normalize = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=(0, 0, 0), std=(1, 1, 1))
        ])

        self.global_transform1 = T.Compose([
            T.RandomResizedCrop(224, global_image_scale, interpolation=InterpolationMode.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(1.0),
            normalize
        ])

        self.global_transform2 = T.Compose([
            T.RandomResizedCrop(224, global_image_scale, interpolation=InterpolationMode.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(p=0.1),
            utils.Solarize(p=0.2),
            normalize
        ])

        self.num_local_crops = num_local_crops

        self.local_crops_transform = T.Compose([
            T.RandomResizedCrop(96, local_image_scale, interpolation=InterpolationMode.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(p=0.5),
            normalize
        ])

    def __call__(self, img):
        crops = [self.global_transform1(img), self.global_transform2(img)]

        for _ in range(self.num_local_crops):
            crops.append(self.local_crops_transform(img))
        return crops


class DinoLoss(nn.Module):
    def __init__(self, out_dim, n_crops, warmup_teacher_temp, teacher_temp, warmup_teacher_epochs,
                 epochs, student_temp=0.1, center_momentum=0.9):
        super(DinoLoss, self).__init__()
        self.n_crops = n_crops
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))
        self.teacher_temp_schedule = np.concatenate(
            [np.linspace(warmup_teacher_temp, teacher_temp, warmup_teacher_epochs),
             (np.ones(epochs - warmup_teacher_epochs) * teacher_temp)])

    def forward(self, student_output, teacher_output, epoch):
        student_output = student_output / self.student_temp
        student_output = student_output.chunk(self.n_crops)

        teacher_output = (teacher_output - self.center) / self.teacher_temp_schedule[epoch]
        teacher_output = F.softmax(teacher_output, dim=-1).chunk(2)

        total_loss = 0
        n_loss_terms = 0
        for t_idx, t in enumerate(teacher_output):
            for s_idx, s in enumerate(student_output):
                if t_idx == s_idx:
                    continue
                loss = torch.mean(-t * F.log_softmax(s, dim=-1), dim=-1)
                total_loss += loss
                n_loss_terms += 1
        total_loss /= n_loss_terms
        self.update_center(teacher_output)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        dist.all_reduce(batch_center)
        batch_center = batch_center / (len(teacher_output) * dist.get_world_size())

        self.center = self.center_momentum * self.center + (1 - self.center_momentum) * batch_center


def train(args):
    utils.init_distributed_mode(args)

    augmentations = DataAugmentationsDINO(
        args.global_crops_scale, args.local_crops_scale, args.num_local_crops)

    dataset = Vis3xDataset(args.data_path, augmentations)

    sampler = DistributedSampler(dataset, shuffle=True)
    data_loader = DataLoader(
        dataset, batch_size=args.batch_size_per_gpu,
        sampler=sampler, num_workers=args.num_workers, pin_memory=True, drop_last=True)

    # args.arch -> ViT architecture to use [vit_tiny, vit_small, vit_base]
    if args.arch in vit.__dict__.keys():
        student = vit.__dict__[args.arch](args.patch_size, drop_path_rate=args.drop_path_rate)
        teacher = vit.__dict__[args.arch](args.patch_size)
        embed_dim = student.embed_dim
    else:
        print(f"Unknown ViT architecture: {args.arch}")
        sys.exit(1)

    # Setup Model
    student = utils.MultiCropWrapper(student, DINOHead(
        embed_dim, args.out_dim, use_bn=args.use_bn_in_head, norm_last_layer=args.norm_last_layer))

    teacher = utils.MultiCropWrapper(teacher, DINOHead(
        embed_dim, args.out_dim, use_bn=args.use_bn_in_head, norm_last_layer=args.norm_last_layer))

    # Send to GPU
    student, teacher = student.cuda(), teacher.cuda()

    if utils.has_batch_norm(student):
        student = nn.SyncBatchNorm.convert_sync_batchnorm(student)
        teacher = nn.SyncBatchNorm.convert_sync_batchnorm(teacher)

        teacher = nn.parallel.DistributedDataParallel(teacher, device_ids=[args.gpu])
        teacher_without_ddp = teacher.module
    else:
        teacher_without_ddp = teacher

    student = nn.parallel.DistributedDataParallel(student, device_ids=[args.gpu])

    teacher_without_ddp.load_state_dict(student.module.state_dict())
    for param in teacher.parameters():
        param.requires_grad = False

    # DINO Loss and send to GPU
    dino_loss = DinoLoss(args.out_dim, args.num_local_crops + 2,
                         args.warmup_teacher_temp, args.teacher_temp,
                         args.warmup_teacher_temp_epochs, args.epochs).cuda()

    param_groups = utils.get_param_groups(student)

    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(param_groups)
    else:
        optimizer = torch.optim.SGD(param_groups, lr=0, momentum=0.9)

    # Whether or not we use fp_16 GradScaler
    scaler = None
    if args.use_fp16:
        scaler = torch.cuda.amp.GradScaler()

    # Create schedulers for weight_decay, lr and teacher momentum
    lr_schedule = utils.cosine_scheduler(
        base_val=args.lr * (args.batch_size_per_gpu * utils.get_world_size()) / 256.,
        final_val=args.min_lr, warmup_epochs=args.lr_warmup_epochs,
        epochs=args.epochs, n_iters_per_epoch=(len(data_loader))
    )
    wd_schedule = utils.cosine_scheduler(
        base_val=args.weight_decay, final_val=args.weight_decay_end,
        epochs=args.epochs, n_iters_per_epoch=len(data_loader)
    )
    momentum_schedule = utils.cosine_scheduler(
        base_val=args.momentum_teacher, final_val=1,
        epochs=args.epochs, n_iters_per_epoch=len(data_loader)
    )
    print(f"Loss, Optimizers and Schedulers ready.")

    restore_point = {"epoch": 0}
    utils.restart_from_ckpt(
        ckpt_path=os.path.join(args.ckpt_dir),
        restore_point=restore_point,
        student=student, teacher=teacher,
        optimizer=optimizer, scaler=scaler, loss=dino_loss
    )

    # Create checkpoint directory if it does not exist
    utils.create_ckpt_dir(args.ckpt_dir)

    min_loss = math.inf
    start_epoch = restore_point["epoch"]

    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            args, epoch, student, teacher, teacher_without_ddp,
            dino_loss, optimizer, scaler, data_loader,
            lr_schedule, wd_schedule, momentum_schedule
        )
        save_dict = {
            "student": student.state_dict(),
            "teacher": teacher.state_dict(),
            "optimizer": optimizer.state_dict(),
            "loss": dino_loss.state_dict()
        }
        if scaler is not None:
            save_dict["scaler"] = scaler.state_dict()

        step_loss = train_stats['loss']

        if args.save_best_only:
            if step_loss < min_loss:
                print(f"Best new min loss: {step_loss}. Creating checkpoint")
                utils.save_master(save_dict, os.path.join(args.ckpt_dir, f"checkpoint-{step_loss}-{epoch:04}.pth"))
                min_loss = step_loss

        if not args.save_best_only and args.ckpt_freq and epoch % args.ckpt_freq == 0:
            utils.save_master(save_dict, os.path.join(args.ckpt_dir, f"checkpoint-{step_loss}-{epoch:04}.pth"))


def train_one_epoch(args, epoch, student, teacher, teacher_without_ddp, dino_loss, optimizer,
                    scaler, data_loader, lr_schedule, wd_schedule, momentum_schedule):
    loop = tqdm(data_loader, desc="DINO")
    for it, images in enumerate(loop):
        it = len(data_loader) * epoch + it

        for idx, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it]
            if idx == 0:
                param_group["weight_decay"] = wd_schedule[it]

        images = [img.cuda(non_blocking=True) for img in images]

        with torch.cuda.amp.autocast(scaler is not None):
            teacher_output = teacher(images[:2])
            student_output = student(images)
            loss = dino_loss(student_output, teacher_output, epoch)

        if not math.isfinite(loss.item()):
            print(f"Loss is not finite: {loss.item()}")
            sys.exit(1)

        optimizer.zero_grad()
        if scaler is None:
            loss.backward()
            if args.clip_grad:
                utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student, args.freeze_last_layer)
            optimizer.step()
        else:
            scaler.scale(loss).backward()
            if args.clip_grad:
                scaler.unscale_(optimizer)
                utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student, args.freeze_last_layer)

            scaler.scale(optimizer).step()
            scaler.update()

        # EMA update for the teacher
        with torch.no_grad():
            m = momentum_schedule[it]  # momentum parameter
            for param_q, param_k in zip(student.module.parameters(), teacher_without_ddp.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

        lr = optimizer.param_groups[0]["lr"],
        wd = optimizer.param_groups[0]["weight_decay"]

        loop.set_postfix(loss=loss.item(), lr=lr, weight_decay=wd)

        return {"loss": loss.item(), "lr": lr, "weight_decay": wd}


def get_args_parser():
    # Copy and Paste From DINO repository
    parser = argparse.ArgumentParser('Vis3x', add_help=False)

    # Model parameters
    parser.add_argument('--arch', default='vit_small', type=str,
                        choices=['vit_tiny', 'vit_small', 'vit_base'],
                        help="""Name of architecture to train. For quick experiments with ViTs,
                        we recommend using vit_tiny or vit_small.""")
    parser.add_argument('--patch_size', default=16, type=int, help="""Size in pixels
        of input square patches - default 16 (for 16x16 patches). Using smaller
        values leads to better performance but requires more memory. Applies only
        for ViTs (vit_tiny, vit_small and vit_base). If <16, we recommend disabling
        mixed precision training (--use_fp16 false) to avoid unstabilities.""")
    parser.add_argument('--out_dim', default=65536, type=int, help="""Dimensionality of
        the DINO head output. For complex and large datasets large values (like 65k) work well.""")
    parser.add_argument('--norm_last_layer', default=True, type=utils.bool_flag,
                        help="""Whether or not to weight normalize the last layer of the DINO head.
        Not normalizing leads to better performance but can make the training unstable.
        In our experiments, we typically set this paramater to False with vit_small and True with vit_base.""")
    parser.add_argument('--momentum_teacher', default=0.996, type=float, help="""Base EMA
        parameter for teacher update. The value is increased to 1 during training with cosine schedule.
        We recommend setting a higher value with small batches: for example use 0.9995 with batch size of 256.""")
    parser.add_argument('--use_bn_in_head', default=False, type=utils.bool_flag,
                        help="Whether to use batch normalizations in projection head (Default: False)")

    # Temperature teacher parameters
    parser.add_argument('--warmup_teacher_temp', default=0.04, type=float,
                        help="""Initial value for the teacher temperature: 0.04 works well in most cases.
        Try decreasing it if the training loss does not decrease.""")
    parser.add_argument('--teacher_temp', default=0.04, type=float, help="""Final value (after linear warmup)
        of the teacher temperature. For most experiments, anything above 0.07 is unstable. We recommend
        starting with the default value of 0.04 and increase this slightly if needed.""")
    parser.add_argument('--warmup_teacher_temp_epochs', default=0, type=int,
                        help='Number of warmup epochs for the teacher temperature (Default: 30).')

    # Training/Optimization parameters
    parser.add_argument('--use_fp16', type=utils.bool_flag, default=True, help="""Whether or not
        to use half precision for training. Improves training time and memory requirements,
        but can provoke instability and slight decay of performance. We recommend disabling
        mixed precision if the loss is unstable, if reducing the patch size or if training with bigger ViTs.""")
    parser.add_argument('--weight_decay', type=float, default=0.04, help="""Initial value of the
        weight decay. With ViT, a smaller value at the beginning of training works well.""")
    parser.add_argument('--weight_decay_end', type=float, default=0.4, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")
    parser.add_argument('--clip_grad', type=float, default=3.0, help="""Maximal parameter
        gradient norm if using gradient clipping. Clipping with norm .3 ~ 1.0 can
        help optimization for larger ViT architectures. 0 for disabling.""")
    parser.add_argument('--batch_size_per_gpu', default=64, type=int,
                        help='Per-GPU batch-size : number of distinct images loaded on one GPU.')
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs of training.')
    parser.add_argument('--freeze_last_layer', default=1, type=int, help="""Number of epochs
        during which we keep the output layer fixed. Typically doing so during
        the first epoch helps training. Try increasing this value if the loss does not decrease.""")
    parser.add_argument("--lr", default=0.0005, type=float, help="""Learning rate at the end of
        linear warmup (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.""")
    parser.add_argument("--lr_warmup_epochs", default=10, type=int,
                        help="Number of epochs for the linear learning-rate warm up.")
    parser.add_argument('--min_lr', type=float, default=1e-6, help="""Target LR at the
        end of optimization. We use a cosine LR schedule with linear warmup.""")
    parser.add_argument('--optimizer', default='adamw', type=str,
                        choices=['adamw', 'sgd', 'lars'],
                        help="""Type of optimizer. We recommend using adamw with ViTs.""")
    parser.add_argument('--drop_path_rate', type=float, default=0.1, help="stochastic depth rate")

    # Multi-crop parameters
    parser.add_argument('--global_crops_scale', type=float, nargs='+', default=(0.4, 1.),
                        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for large global view cropping. When disabling multi-crop (--local_crops_number 0), we
        recommend using a wider range of scale ("--global_crops_scale 0.14 1." for example)""")
    parser.add_argument('--num_local_crops', type=int, default=8, help="""Number of small
        local views to generate. Set this parameter to 0 to disable multi-crop training.
        When disabling multi-crop we recommend to use "--global_crops_scale 0.14 1." """)
    parser.add_argument('--local_crops_scale', type=float, nargs='+', default=(0.05, 0.4),
                        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for small local view cropping of multi-crop.""")
    
    
    # Misc
    parser.add_argument('--data_path', default='/storage/PCB/train/', type=str,
                        help='Please specify path to the ImageNet training data.')
    parser.add_argument('--ckpt_dir', default="./vis3x_checkpoints/", type=str,
                        help='Path to save logs and checkpoints.')
    parser.add_argument('--save_best_only', default=False, type=utils.bool_flag,
                        help="""Whether or not to save the best model when epochs are rolling""")
    parser.add_argument('--ckpt_freq', default=20, type=int, help='Save checkpoint every x epochs.')
    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    parser.add_argument('--num_workers', default=8,type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")

    return parser


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Vis3x", parents=[get_args_parser()])
    args = parser.parse_args()
    train(args)
