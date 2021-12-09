import argparse
import colorsys
import os
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
import skimage.io as io
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.utils
from PIL import Image
from matplotlib.patches import Polygon
from skimage.measure import find_contours

import vision_transformers as vit


def apply_mask(image, mask, color, alpha=0.5):
    for c in range(3):
        image[:, :, c] = image[:, :, c] * (1 - alpha * mask) + alpha * mask * color[c] * 255
    return image


def random_colors(N, bright=True):
    """
    Generate random colors.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


def display_instances(image, mask, fname="test", figsize=(5, 5), blur=False, contour=True, alpha=0.5):
    fig = plt.figure(figsize=figsize, frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax = plt.gca()

    N = 1
    mask = mask[None, :, :]
    # Generate random colors
    colors = random_colors(N)

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    margin = 0
    ax.set_ylim(height + margin, -margin)
    ax.set_xlim(-margin, width + margin)
    ax.axis('off')
    masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        color = colors[i]
        _mask = mask[i]
        if blur:
            pass
            # _mask = cv2.blur(_mask, (10, 10))
        # Mask
        masked_image = apply_mask(masked_image, _mask, color, alpha)
        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        if contour:
            padded_mask = np.zeros((_mask.shape[0] + 2, _mask.shape[1] + 2))
            padded_mask[1:-1, 1:-1] = _mask
            contours = find_contours(padded_mask, 0.5)
            for verts in contours:
                # Subtract the padding and flip (y, x) to (x, y)
                verts = np.fliplr(verts) - 1
                p = Polygon(verts, facecolor="none", edgecolor=color)
                ax.add_patch(p)
    ax.imshow(masked_image.astype(np.uint8), aspect='auto')
    fig.savefig(fname)
    print(f"{fname} saved.")
    return


@torch.inference_mode()
def visualize(args):
    model = vit.__dict__[args.arch](args.patch_size, n_classes=0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    model.eval()

    if args.ckpt_file_path is None:
        print(f"Pass checkpoint file path to load model checkpoint.")
        sys.exit(1)

    if not os.path.exists(args.ckpt_file_path):
        print(f"Checkpoint file path: {args.ckpt_file_path} does not exist.")
        if not os.path.isfile(args.ckpt_file_path):
            print(f"Checkpoint file path: {args.ckpt_file_path} is not a file.")

    state_dict = torch.load(args.ckpt_file_path, map_location="cpu")

    if args.ckpt_key is not None and args.ckpt_key in state_dict:
        print(f"Take key {args.ckpt_key} in provided checkpoint dict")
        state_dict = state_dict[args.ckpt_key]

    # remove `module.` and `backbone.` prefix added by MultiCropWrapper
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}

    msg = model.load_state_dict(state_dict, strict=False)
    print(f"Loaded pretrained weights from {args.ckpt_file_path} with {msg}")

    if args.image_path is None or not os.path.isfile(args.image_path):
        print(f"Path: {args.image_path} is not empty or is not a file.")

    # Open image
    with open(args.image_path, "rb") as file:
        img = Image.open(file).convert("RGB")

    # Apply transform to image
    transforms = T.Compose([
        T.Resize(args.image_size),
        T.ToTensor(),
        T.Normalize(mean=(0.4267, 0.4158, 0.3837), std=(0.3113, 0.2909, 0.2779))
    ])

    img = transforms(img)

    # Make image divisible by patch size and select that portion in the img
    w, h = img.shape[1] - img.shape[1] % args.patch_size, img.shape[2] - img.shape[2] % args.patch_size
    img = img[:, :w, :h].unsqueeze(0)

    # Width & Height of patch
    w_feature_map = w // args.patch_size
    h_feature_map = h // args.patch_size

    attentions = model.get_last_self_attn(img.to(device))  # b, heads, output, num_patches + 1

    num_heads = attentions.shape[1]
    attentions = attentions[0, :, 0, 1:].reshape(num_heads, -1)  # heads, num_patches

    if args.threshold is not None:
        # we keep only a certain percentage of the mass
        val, idx = torch.sort(attentions)
        # make values sum to 1
        val /= torch.sum(val, dim=1, keepdim=True)
        cumval = torch.cumsum(val, dim=1)
        th_attn = cumval > (1 - args.threshold)
        # Not sure what below 3 lines are doing
        # sort the indices of where those values used to be before sorting

        idx2 = torch.argsort(idx)
        for head in range(num_heads):
            th_attn[head] = th_attn[head][idx2[head]]
        th_attn = th_attn.reshape(num_heads, w_feature_map, h_feature_map).float()
        # interpolate
        th_attn = F.interpolate(th_attn.unsqueeze(0), scale_factor=args.patch_size, mode="nearest")
        th_attn = th_attn[0].cpu().numpy()

    attentions = attentions.reshape(num_heads, w_feature_map, h_feature_map).float()
    attentions = F.interpolate(attentions.unsqueeze(0), scale_factor=args.patch_size, mode="nearest")
    attentions = attentions[0].cpu().numpy()

    os.makedirs(args.output_dir, exist_ok=True)

    torchvision.utils.save_image(torchvision.utils.make_grid(img, normalize=True, scale_each=True),
                                 os.path.join(args.output_dir, "image.png"))

    for h in range(num_heads):
        img_path = os.path.join(args.output_dir, f"attn_head_{h}.png")
        plt.imsave(img_path, attentions[h], format="png")
        print(f"Attention image saved to: {img_path}")

    if args.threshold is not None:
        image = io.imread(os.path.join(args.output_dir, "image.png"))
        for h in range(num_heads):
            mask_th_name = os.path.join(args.output_dir, f"mask_th{args.threshold}_head{h}.png")
            display_instances(image, th_attn[h], fname=mask_th_name, blur=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Vis3x visualize self-attention maps')
    parser.add_argument('--arch', default='vit_small', type=str,
                        choices=['vit_tiny', 'vit_small', 'vit_base'], help='Architecture (support only ViT atm).')
    parser.add_argument('--patch_size', default=16, type=int, help='Patch resolution of the model.')
    parser.add_argument('--ckpt_file_path', default='/notebooks/vis3x/vis3x_checkpoints/checkpoint.pth', type=str,
                        help="Path to pretrained weights to load.")
    parser.add_argument("--ckpt_key", default="teacher", type=str,
                        help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument("--image_path", default="/notebooks/vis3x/pcb_test_49.jpeg", type=str,
                        help="Path of the image to load.")
    parser.add_argument("--image_size", default=(224, 224), type=int, nargs="+", help="Resize image.")
    parser.add_argument('--output_dir', default='./attention-maps', help='Path where to save visualizations.')
    parser.add_argument("--threshold", type=float, default=None, help="""We visualize masks
           obtained by thresholding the self-attention maps to keep xx% of the mass.""")
    args = parser.parse_args()

    visualize(args)
