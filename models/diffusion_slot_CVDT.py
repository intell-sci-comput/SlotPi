# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

# 这个包是创建一个优化 statm预测结果的 基于slot的模型
import torch
import torch.nn as nn
import numpy as np
import math
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp
from einops import rearrange, reduce, repeat
import torch.nn.functional as F
def modulate(x, shift, scale, T):

    N, M = x.shape[-2], x.shape[-1] # N: number of patches, M: hidden size
    B = scale.shape[0]
    x = rearrange(x, '(b t) n m-> b (t n) m',b=B,t=T,n=N,m=M)\
   
    x = x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
    x = rearrange(x, 'b (t n) m-> (b t) n m',b=B,t=T,n=N,m=M)
    return x

#-------------------------------------------------------------------------------
# Positional Encoding For Slots
#-------------------------------------------------------------------------------
def get_sin_pos_enc(seq_len, d_model):
    """Sinusoid absolute positional encoding."""
    inv_freq = 1. / (10000**(torch.arange(0.0, d_model, 2.0) / d_model))
    pos_seq = torch.arange(seq_len - 1, -1, -1).type_as(inv_freq)
    sinusoid_inp = torch.outer(pos_seq, inv_freq)
    pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)
    return pos_emb.unsqueeze(0)  # [1, L, C]


def build_pos_enc(pos_enc, input_len, d_model):
    """Positional Encoding of shape [1, L, D]."""
    if not pos_enc:
        return None
    # ViT, BEiT etc. all use zero-init learnable pos enc
    if pos_enc == 'learnable':
        pos_embedding = nn.Parameter(torch.zeros(1, input_len, d_model))
    # in SlotFormer, we find out that sine P.E. is already good enough
    elif 'sin' in pos_enc:  # 'sin', 'sine'
        pos_embedding = nn.Parameter(
            get_sin_pos_enc(input_len, d_model), requires_grad=False)
    else:
        raise NotImplementedError(f'unsupported pos enc {pos_enc}')
    return pos_embedding

#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings



def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

#################################################################################
#                                 Core VDT Model                                #
#################################################################################

class VDTBlock(nn.Module):
    """
    A VDT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, mode='video', num_frames=16, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)

      
        # approx_gelu = lambda: nn.GELU(approximate="tanh")
        # self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)

        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, drop=0)
        
      
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

        self.num_frames = num_frames
        
        self.mode = mode
        
        ## Temporal Attention Parameters
        if self.mode == 'video':
            
            self.temporal_norm1 = nn.LayerNorm(hidden_size)
            self.temporal_attn = Attention(
              hidden_size, num_heads=num_heads, qkv_bias=True)
            self.temporal_fc = nn.Linear(hidden_size, hidden_size)

    def forward(self, x, c):

        # shift_msa shape: (B, hidden_size) 
       
        # shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        T = self.num_frames
        K, N, M = x.shape # B*T,P,D
        B = K // T
        if self.mode == 'video':
            
           
            x = rearrange(x, '(b t) n m -> (b n) t m',b=B,t=T,n=N,m=M)
            res_temporal = self.temporal_attn(self.temporal_norm1(x))
            res_temporal = rearrange(res_temporal, '(b n) t m -> (b t) n m',b=B,t=T,n=N,m=M)
            res_temporal = self.temporal_fc(res_temporal)
            x = rearrange(x, '(b n) t m -> (b t) n m',b=B,t=T,n=N,m=M)
            x = x + res_temporal

       
       
        attn = self.attn(modulate(self.norm1(x), shift_msa, scale_msa, self.num_frames))
        attn = rearrange(attn, '(b t) n m-> b (t n) m',b=B,t=T,n=N,m=M)
        attn = gate_msa.unsqueeze(1) * attn
        attn = rearrange(attn, 'b (t n) m-> (b t) n m',b=B,t=T,n=N,m=M)
        x = x + attn

        mlp = self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp, self.num_frames))
        mlp = rearrange(mlp, '(b t) n m-> b (t n) m',b=B,t=T,n=N,m=M)
        mlp = gate_mlp.unsqueeze(1) * mlp
        mlp = rearrange(mlp, 'b (t n) m-> (b t) n m',b=B,t=T,n=N,m=M)
        x = x + mlp


        return x


class VDTBlock1(nn.Module):
    """
    A VDT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    time att + space att
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, mode='video', num_frames=16, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)

        
        # approx_gelu = lambda: nn.GELU(approximate="tanh")
        # self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)

        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, drop=0)
        
        
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

        self.num_frames = num_frames
        
        self.mode = mode
        
        ## Temporal Attention Parameters
        if self.mode == 'video':
            
            self.temporal_norm1 = nn.LayerNorm(hidden_size)
            self.temporal_attn = Attention(
              hidden_size, num_heads=num_heads, qkv_bias=True)
            self.temporal_fc = nn.Linear(hidden_size, hidden_size)

    def forward(self, x, c):

        # shift_msa shape: (B, hidden_size) 
       
       
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        T = self.num_frames
        K, N, M = x.shape # B*T,P,D
        B = K // T
        if self.mode == 'video':
            
            xt = rearrange(x, '(b t) n m -> (b n) t m',b=B,t=T,n=N,m=M)
            res_temporal = self.temporal_attn(self.temporal_norm1(xt))
            res_temporal = rearrange(res_temporal, '(b n) t m -> (b t) n m',b=B,t=T,n=N,m=M)
            res_temporal = self.temporal_fc(res_temporal)
            xt = rearrange(xt, '(b n) t m -> (b t) n m',b=B,t=T,n=N,m=M)
            xt = xt + res_temporal

        # 空间注意力 
       
        attn = self.attn(modulate(self.norm1(x), shift_msa, scale_msa, self.num_frames))
        attn = rearrange(attn, '(b t) n m-> b (t n) m',b=B,t=T,n=N,m=M)
        attn = gate_msa.unsqueeze(1) * attn
        attn = rearrange(attn, 'b (t n) m-> (b t) n m',b=B,t=T,n=N,m=M)
        xs = x + attn

        # 融合
        x = xs + xt

        mlp = self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp, self.num_frames))
        mlp = rearrange(mlp, '(b t) n m-> b (t n) m',b=B,t=T,n=N,m=M)
        mlp = gate_mlp.unsqueeze(1) * mlp
        mlp = rearrange(mlp, 'b (t n) m-> (b t) n m',b=B,t=T,n=N,m=M)
        x = x + mlp


        return x

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class FinalLayer(nn.Module):
    """
    The final layer of VDT.
    """
    def __init__(self, hidden_size, patch_size, out_channels, num_frames):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )
        self.num_frames = num_frames

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1) # (B, 2*hidden_size) -> (B, hidden_size)
        x = modulate(self.norm_final(x), shift, scale, self.num_frames)
        x = self.linear(x)
        return x

class Slot_FinalLayer(nn.Module):
    """
    The final layer of VDT.
    """
    def __init__(self, hidden_size, num_frames):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, hidden_size, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )
        self.num_frames = num_frames

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale, self.num_frames)
        x = self.linear(x)
        return x


class VDT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=25,
        num_heads=8,
        mlp_ratio=4.0,
        # class_dropout_prob=0.1,
        # num_classes=1000,
        learn_sigma=True,
        mode='video',
        num_frames=16,
        slot_size = 128,
        num_slot = 7
    ):
        super().__init__()
        # self.learn_sigma = learn_sigma
        # self.in_channels = in_channels
        # self.out_channels = in_channels * 2 if learn_sigma else in_channels
        # self.patch_size = patch_size
        self.num_heads = num_heads
        self.slot_size = slot_size
        self.num_slot = num_slot
        self.num_frames = num_frames
        # print(depth)
        # self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
       
        self.x_embedder = nn.Linear(self.slot_size*2, slot_size*2)

      
        # self.t_embedder = TimestepEmbedder(hidden_size)
        self.t_embedder = TimestepEmbedder(self.slot_size*2)
        
        self.mode = mode
        
        if self.mode == 'video':
            self.pos_t_embed = build_pos_enc(pos_enc='sin',input_len=self.num_frames, d_model=self.slot_size*2) # (1, T, D)
            self.time_drop = nn.Dropout(p=0)
        self.pos_s_embed = build_pos_enc(pos_enc='sin',input_len=self.num_slot, d_model=self.slot_size*2) # (1, P, D)
       
      
       
        self.blocks = nn.ModuleList([
            VDTBlock(self.slot_size*2, 
                     self.num_heads, 
                     mlp_ratio=mlp_ratio, 
                     mode=mode,
                     num_frames=self.num_frames) 
                       for _ in range(depth)])

        
        # self.final_layer = FinalLayer(slot_size*2, self.num_frames)
        # self.initialize_weights()      
        # self.out_conv = nn.Conv2d(self.out_channels,64,1)
        self.slt_final_layer = Slot_FinalLayer(self.slot_size*2, self.num_frames)
        self.initialize_weights()

        # approx_gelu = lambda: nn.GELU()
        self.ffc = Mlp(in_features=self.slot_size*2, hidden_features=self.slot_size*8,
                       out_features=self.slot_size, drop=0.)

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        # pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        # self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
  

        # if self.mode == 'video':
        #     grid_num_frames = np.arange(self.num_frames, dtype=np.float32)
        #     time_embed = get_1d_sincos_pos_embed_from_grid(self.pos_embed.shape[-1], grid_num_frames)
        #     self.time_embed.data.copy_(torch.from_numpy(time_embed).float().unsqueeze(0))

        # Initialize input_embed like nn.Linear (instead of nn.Conv2d):
        nn.init.xavier_uniform_(self.x_embedder.weight)  # 初始化权重
        nn.init.constant_(self.x_embedder.bias, 0)  # 初始化偏置为0

        # Initialize label embedding table:
        # nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in VDT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.slt_final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.slt_final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.slt_final_layer.linear.weight, 0)
        nn.init.constant_(self.slt_final_layer.linear.bias, 0)

    def forward(self, x, t):
        """
        Forward pass of VDT.
        x: (N, T,C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        # print("depth is:",self.depth)
        # B, T, C, W, H = x.shape 
        B, T , N , D, C = x.shape
        assert T == self.num_frames and N ==self.num_slot

        x = x.contiguous().view(B, T, N, -1) # (B, T, N, D, C) -> (B, T, N, D)
        # x = x.contiguous().view(-1, C, W, H)

       
        x = self.x_embedder(x)  #(B, T, N, D) 
        
       
        slots_s_pe = self.pos_s_embed.unsqueeze(1).\
                repeat(B, self.num_frames, 1, 1)#(1,N,D)->(1,1,N,D)->(B,T,N,D)
        x = x + slots_s_pe # (B, T, N, D)

        if self.mode == 'video':
           
            slot_t_pe = self.pos_t_embed.unsqueeze(2).\
            repeat(B, 1, self.num_slot, 1) #(1,T,D)->(1,T,1,D)->(B,T,N,D)

            x = x + slot_t_pe
            x = self.time_drop(x)
            x = rearrange(x, 'b t n m -> (b t) n m',b=B,t=T)
        
        t = self.t_embedder(t)                   # (B, D)
        # y = self.y_embedder(y, self.training)    # (B, D)

        # c = t + y                             # (B, D)
        c = t
        for block in self.blocks:
            x = block(x, c)                      # (B, P, D)
        
        
        # x = self.final_layer(x, c)                # (N, T, patch_size ** 2 * out_channels)
        # x = self.unpatchify(x)                   # (N, out_channels, H, W)
        # x = self.out_conv(x)
        # x = x.view(B, T, x.shape[-3], x.shape[-2], x.shape[-1])
        # 基于slot的final_layer修改
        x = self.slt_final_layer(x, c) # (B, T, N, 2*D)
        x = self.ffc(x) # (B*T, N, D)
        x = x.view(B, T, N, D) # (B, T, N, D)
        return x

    def forward_with_cfg(self, x, t, y, cfg_scale):
        """
        Forward pass of VDT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y)
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


# #################################################################################
# #                                   VDT Configs                                  #
# #################################################################################

# def VDT_L_2(**kwargs):
#     return VDT(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)

# def VDT_S_2(**kwargs):
#     return VDT(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)


# VDT_models = {
#     'VDT-L/2':  VDT_L_2,  
#     'VDT-S/2':  VDT_S_2, 
# }




import torch.nn.functional as F


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)

def quadratic_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2

def sigmoid_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    betas = torch.linspace(-6, 6, timesteps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start






# timesteps = 300
timesteps = 1000
# define beta schedule
betas = linear_beta_schedule(timesteps=timesteps)

# define alphas 
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

# calculations for diffusion q(x_t | x_{t-1}) and others
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

# calculations for posterior q(x_{t-1} | x_t, x_0)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)



# forward diffusion (using the nice property)
def q_sample(x_start, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x_start)

    sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, x_start.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x_start.shape
    )

    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise




def p_losses(denoise_model, x_start, x_pre,t,x_c, noise=None, loss_type="l1"):
    '''
    Denoising loss for VDT.
    Args:
   
    '''

    x_start = x_start.unsqueeze(-1)
    x_pre = x_pre.unsqueeze(-1)
    x_c = x_c.unsqueeze(-1)
    if noise is None:
        noise = torch.randn_like(x_start)
    
    x_C = torch.cat([x_c,x_c],-1)
    x_noisy = q_sample(x_start=x_start, t=t, noise=noise)
    x_noisy = torch.cat([x_noisy,x_pre],-1)
    
    x_noisy = torch.cat([x_C,x_noisy],1)
   
    predicted_noise = denoise_model(x_noisy, t)[:,x_c.shape[1]:,]

    noise = noise.squeeze(-1)
    if loss_type == 'l1':
        loss = F.l1_loss(noise.squeeze(), predicted_noise)
    elif loss_type == 'l2':
        loss = F.mse_loss(noise, predicted_noise)
    elif loss_type == "huber":
        loss = F.smooth_l1_loss(noise, predicted_noise)
    else:
        raise NotImplementedError()

    return loss

@torch.no_grad()
def p_sample(model, x, x_pre,x_c,t, t_index):
    betas_t = extract(betas, t, x.shape) # (B, 1)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x.shape
    )
    sqrt_recip_alphas_t = extract(sqrt_recip_alphas, t, x.shape)
    
    # Equation 11 in the paper
    # Use our model (noise predictor) to predict the mean\
    # 采样的过程中 x = noise = x_start
    x = x.unsqueeze(-1) # (B, T, N, D) -> (B, T1, N, D, 1)
    x_pre = x_pre.unsqueeze(-1)
    x_c = x_c.unsqueeze(-1)

    x_noise = torch.cat([x,x_pre],-1) # (B, T2, N, D, 2)
    x_C = torch.cat([x_c,x_c],-1) # (B, T1, N, D, 2)
    x_noise = torch.cat([x_C,x_noise],1) # (B, T1+T2, N, D, 2)
    noise1 = model(x_noise, t)[:,x_c.shape[1]:,] # (B, T, N, D)


    model_mean = sqrt_recip_alphas_t * (
        x.squeeze(-1) - betas_t * noise1 / sqrt_one_minus_alphas_cumprod_t
    )

    if t_index == 0:
        return model_mean
    else:
        posterior_variance_t = extract(posterior_variance, t,  x.squeeze(-1).shape)
        noise = torch.randn_like(x.squeeze(-1))
        # Algorithm 2 line 4:
        return model_mean + torch.sqrt(posterior_variance_t) * noise 

# Algorithm 2 (including returning all images)
@torch.no_grad()
def p_sample_loop(model,x_pre,x_c,shape):
    device = next(model.parameters()).device

    b = shape[0]
    # start from pure noise (for each example in the batch)

    # slots = torch.randn(shape, device=device)
    slots = torch.randn_like(x_pre, device=device)

    # imgs = []

    # timesteps = torch.tensor([50],device=device)
    # img = q_sample(x_pre,timesteps)

    # for i in tqdm(reversed(range(0, timesteps)), desc='sampling loop time step', total=timesteps):
    # for i in reversed(range(0, timesteps.item())):


    for i in reversed(range(0, timesteps)):
        slots = p_sample(model, slots, x_pre,x_c,torch.full((b,), i, device=device, dtype=torch.long), i)
        # imgs.append(img.cpu().numpy())
    return slots

@torch.no_grad()
def sample(model,x_pre,x_c, slot_size= 128,num_slot= 7, batch_size=16, time = 12):
    '''
    '''
    assert x_pre.shape[1] == time
    return p_sample_loop(model,x_pre,x_c, shape=(batch_size, time, num_slot, slot_size))


