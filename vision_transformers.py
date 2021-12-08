import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import  InterpolationMode


def drop_path(x, drop_prob: float, training: bool = False):
    if training:
        return x
    # drop_path = x / keep_prob * (keep_prob + random_tensor)
    keep_probability = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndims - 1)
    random_tensor = keep_probability + torch.randn(shape, device=x[0].device)
    random_tensor.floor_()
    return x.div(keep_probability) * random_tensor


class DropPath(nn.Module):
    def __init__(self, p: float):
        super(DropPath, self).__init__()
        self.probability = p

    def forward(self, x):
        x = drop_path(x, self.probability, self.training)
        return x


class PatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size=16, in_dims=3, embed_dim=768):
        super(PatchEmbedding, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.embedding = nn.Conv2d(in_dims, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.embedding(x)  # B, E, W, H
        return x.flatten(2).permute(0, 2, 1)  # B, WxH, E


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_dropout=0., proj_dropout=0.):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = qk_scale or self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.projection = nn.Linear(dim, dim)
        self.projection_dropout = nn.Dropout(proj_dropout)

    def forward(self, x):
        B, N, E = x.shape
        # 3, B, H, N, E
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        k_t = k.transpose(-2, -1)
        attns = (q @ k_t) * self.scale  # N, E @ E, N
        attns = torch.softmax(attns, dim=-1)
        attns = self.attn_dropout(attns)  # N, N

        x = (attns @ v).reshape(B, N, E)  # N, N @  N, E
        x = self.projection(x)
        x = self.projection_dropout(x)

        return x, attns


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, activation=nn.GELU, drop=0., ):
        super(Mlp, self).__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        self.lin1 = nn.Linear(in_features, hidden_features)
        self.activation = activation()
        self.lin2 = nn.Linear(hidden_features, out_features)
        self.dropout = nn.Dropout(drop)

    def forward(self, x):
        x = self.lin1(x)
        x = self.activation(x)
        x = self.lin2(x)
        return self.dropout(x)


class Block(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., drop=0.,
                 drop_path_prob=0., mlp_ratio=4, norm_layer=nn.LayerNorm, activation=nn.GELU):
        super(Block, self).__init__()
        self.norm_layer1 = norm_layer(dim)
        self.attn_layer = Attention(dim, num_heads, qkv_bias, qk_scale, attn_drop, drop)
        self.drop_path = DropPath(drop_path_prob) if drop_path_prob > 0. else nn.Identity()

        self.norm_layer2 = norm_layer(dim)
        self.mlp = Mlp(dim, dim * mlp_ratio, activation=activation, drop=drop)

    def forward(self, x):
        y, attns = self.attn_layer(self.norm_layer1(x))

        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm_layer2(x)))

        return x, attns


class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, n_classes=3, patch_size=16, in_channels=3, embed_dim=768,
                 n_blocks=3, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., drop=0.,
                 drop_path_rate=0., mlp_ratio=4, norm_layer=nn.LayerNorm, activation=nn.GELU, **kwargs):
        super(VisionTransformer, self).__init__()
        self.embed_dim = embed_dim
        self.patch_embedding = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)

        # Learnable class token and positional embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.positional_embedding = nn.Parameter(torch.zeros(1, self.patch_embedding.num_patches + 1, embed_dim))
        self.positional_dropout = nn.Dropout()

        # Stochastic depth update rule
        dropout_path_rates = [x.item() for x in torch.linspace(0, drop_path_rate, n_blocks)]

        self.blocks = nn.ModuleList(
            [
                Block(embed_dim, num_heads, qkv_bias, qk_scale, attn_drop,
                      drop, dropout_path_rates[i], mlp_ratio, norm_layer, activation)
                for i in range(n_blocks)
            ])
        self.norm_layer = norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, n_classes) if n_classes > 0 else nn.Identity()

        nn.init.trunc_normal_(self.cls_token, std=.02)
        nn.init.trunc_normal_(self.positional_embedding, std=.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        # w of Linear = truncated_normal, b of Linear = constant 0
        # w of LayerNorm = constant 1, b of LayerNorm = constant 0
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1)

    def interpolate_pos_embedding(self, x, w, h):
        n_patch = x.shape[1] - 1
        N = self.positional_embedding.shape[1] - 1
        if n_patch == N and w == h:
            return self.positional_embedding

        # Below is from DINO repository
        class_pos_embed = self.positional_embedding[:, 0]
        patch_pos_embed = self.positional_embedding[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_embedding.patch_size
        h0 = h // self.patch_embedding.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = F.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic',
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)

        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def prepare_tokens(self, x):
        b, c, w, h = x.shape
        x = self.patch_embedding(x)  # batch, num_tokens, embedding_dim
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.interpolate_pos_embedding(x, w, h)
        return x

    def forward(self, x):
        x = self.prepare_tokens(x)

        for block in self.blocks:
            x = block(x)[0]

        # Normalize and return cls_token
        x = self.norm_layer(x)

        return x[:, 0]

    def get_last_self_attn(self, x):
        x = self.prepare_tokens(x)
        attns = None
        for idx, block in enumerate(self.blocks):
            x, attns = block(x)
        return attns

    def get_last_n_outputs(self, x, n=1):
        x = self.prepare_tokens(x)
        outputs = []
        for idx, block in enumerate(self.blocks):
            x = block(x)[0]
            if len(self.blocks) - 1 <= n:
                outputs.append(x)
        return outputs


class DINOHead(nn.Module):
    def __init__(self, in_dim, out_dim, n_layers=3, use_bn=False, norm_last_layer=True, hidden_dim=2048,
                 bottleneck_dim=256):
        super(DINOHead, self).__init__()
        self.n_layers = max(n_layers, 1)
        if self.n_layers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())

            for _ in range(self.n_layers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))

            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        x = F.normalize(x, p=2, dim=-1)
        x = self.last_layer(x)
        return x


def vit_tiny(patch_size=16, **kwargs):
    return VisionTransformer(
        patch_size=patch_size, embed_dim=192, n_blocks=12, num_heads=3, mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), qkv_bias=True, **kwargs)


def vit_small(patch_size=16, **kwargs):
    return VisionTransformer(
        patch_size=patch_size, embed_dim=384, n_blocks=12, num_heads=6, mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), qkv_bias=True, **kwargs)


def vit_base(patch_size=16, **kwargs):
    return VisionTransformer(
        patch_size=patch_size, embed_dim=768, n_blocks=12, num_heads=12, mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), qkv_bias=True, **kwargs)


if __name__ == "__main__":
    imgs = torch.randn(10, 3, 224, 224)
    v = VisionTransformer(224)
    output = v.get_last_self_attn(imgs)
    print(f'ViT Output shape {output.shape}')
