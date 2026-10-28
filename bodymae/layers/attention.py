import torch
import torch.nn as nn
from bodymae.layers.mlp import MLPNet
from timm.models.layers import DropPath
import einops
try:
    import flash_attn
except ImportError:
    flash_attn = None


from torch.nn.functional import scaled_dot_product_attention


class AttentionD(nn.Module):
    def __init__(self, channels, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., use_flash_attn=True,
                 proj=False, proj_drop=0., upcast_attention=False, upcast_softmax=False):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.qkv = torch.nn.Linear(channels, channels * 3, bias=qkv_bias)
        self.scale = qk_scale or (channels // num_heads) ** -0.5
        self.attn_drop = attn_drop
        self.use_flash_attn = use_flash_attn
        if use_flash_attn:
            assert flash_attn is not None
        else:
            self.softmax = torch.nn.Softmax(dim=-1)
            self.attn_drop_layer = torch.nn.Dropout(attn_drop)

        self.upcast_attention = upcast_attention
        self.upcast_softmax = upcast_softmax
        if proj:
            self.proj = nn.Sequential(nn.Linear(channels, channels),
                                      nn.Dropout(proj_drop))
        else:
            self.proj = nn.Identity()

    def forward(self, x, cu_seqlens, patch_size=300):



        qkv = self.qkv(x)
        model_type = qkv.dtype

        if self.use_flash_attn:

            x = flash_attn.flash_attn_varlen_qkvpacked_func(qkv.reshape(-1, 3, self.num_heads, self.channels // self.num_heads).half(),
                                                            cu_seqlens,
                                                            max_seqlen=patch_size,
                                                            dropout_p=self.attn_drop if self.training else 0,
                                                            softmax_scale=self.scale,
                                                            ).reshape(-1, self.channels)
            x = x.to(model_type)
        else:


            q, k, v = qkv.reshape(-1, patch_size, 3, self.num_heads, self.channels // self.num_heads).permute(2, 0, 3, 1, 4).unbind(dim=0)
            if self.upcast_attention:
                q = q.float()
                k = k.float()
            attn = (q * self.scale) @ k.transpose(-2, -1)
            if self.upcast_softmax:
                attn = attn.float()
            attn = self.softmax(attn)
            attn = self.attn_drop_layer(attn).to(model_type)
            x = (attn @ v).transpose(1, 2).reshape(-1, self.channels)


        x = self.proj(x)

        return x


class FFN(nn.Module):
    def __init__(self, in_channels, hidden_channels=None, out_channels=None, act_layer=nn.GELU, dropout=0.):
        super().__init__()
        hidden_channels = hidden_channels or in_channels
        out_channels = out_channels or in_channels
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_channels, out_channels)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x


class TransformerBlockD(nn.Module):
    def __init__(self, channels, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., use_flash_attn=True,
                 proj=False, proj_drop=0., upcast_attention=False, upcast_softmax=False,
                 ffn_ratio=4, ffn_act=nn.GELU, ffn_dropout=0.,
                 drop_path=0., norm_layer=nn.LayerNorm,
                 pre_norm=True):
        super().__init__()
        self.norm1 = norm_layer(channels, eps=1e-3)
        self.attn = AttentionD(channels, num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
                               use_flash_attn=use_flash_attn, proj=proj, proj_drop=proj_drop,
                               upcast_attention=upcast_attention, upcast_softmax=upcast_softmax)
        self.norm2 = norm_layer(channels, eps=1e-3)
        self.ffn = FFN(in_channels=channels, hidden_channels=int(channels*ffn_ratio), act_layer=ffn_act, dropout=ffn_dropout)

        self.drop_path = DropPath(drop_path)
        self.pre_norm = pre_norm

    def forward(self, x, cu_seqlens, pos_emb=0, patch_size=300):
        x = x + pos_emb
        shortcut = x
        if self.pre_norm:
            x = self.norm1(x)
        x = self.drop_path(self.attn(x, cu_seqlens, patch_size))


        x = x + shortcut

        if not self.pre_norm:
            x = self.norm1(x)

        shortcut = x
        if self.pre_norm:
            x = self.norm2(x)

        x = self.drop_path(self.ffn(x))

        x = shortcut + x

        if not self.pre_norm:
            x = self.norm2(x)
        return x


class Attention(nn.Module):
    def __init__(self, channels, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., use_flash_attn=True,
                 proj=False, proj_drop=0., upcast=False):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.qkv = torch.nn.Linear(channels, channels * 3, bias=qkv_bias)
        self.scale = qk_scale or (channels // num_heads) ** -0.5
        self.attn_drop = attn_drop
        self.use_flash_attn = use_flash_attn
        if use_flash_attn:
            assert flash_attn is not None
        else:
            self.softmax = torch.nn.Softmax(dim=-1)
            self.attn_drop_layer = torch.nn.Dropout(attn_drop)

        self.upcast = upcast
        if proj:
            self.proj = nn.Sequential(nn.Linear(channels, channels),
                                      nn.Dropout(proj_drop))
        else:
            self.proj = nn.Identity()

    def forward(self, x, mask=None, rpe=None):

        B, N, C = x.shape
        qkv = self.qkv(x)
        model_type = qkv.dtype
        if self.upcast:
            qkv = qkv.float()


        if self.use_flash_attn:
            qkv = qkv.chunk(3, dim=-1)
            q, k, v = map(lambda t: einops.rearrange(t, "b n (h d) -> b h n d", h=self.num_heads), qkv)
            x = scaled_dot_product_attention(query=q, key=k, value=v, attn_mask=mask,
                                             dropout_p=self.attn_drop if self.training else 0,
                                             scale=self.scale)
            x = einops.rearrange(x, "b h n d -> b n (h d)")
        else:
            qkv = qkv.chunk(3, dim=-1)
            q, k, v = map(lambda t: einops.rearrange(t, "b n (h d) -> b h n d", h=self.num_heads), qkv)
            attn = (q * self.scale) @ k.transpose(-2, -1)
            if rpe is not None:
                attn = attn + rpe

            if mask is not None:
                mask = einops.repeat(mask, "b n n -> b h n n", h=self.num_heads)
                attn = attn.masked_fill(mask.logical_not(), float("-inf"))


            attn = self.softmax(attn)
            attn = self.attn_drop_layer(attn)
            x = einops.rearrange(attn @ v, "b h n d -> b n (h d)")

        x = x.to(model_type)
        x = self.proj(x)

        return x


class TransformerBlock(nn.Module):
    def __init__(self, channels, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., use_flash_attn=True,
                 proj=False, proj_drop=0., upcast=False,
                 ffn_ratio=4, ffn_act=nn.GELU, ffn_dropout=0.,
                 drop_path=0., norm_layer=nn.LayerNorm,
                 pre_norm=True):
        super().__init__()
        self.norm1 = norm_layer(channels)
        self.attn = Attention(channels, num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
                              use_flash_attn=use_flash_attn, proj=proj, proj_drop=proj_drop, upcast=upcast)
        self.norm2 = norm_layer(channels)
        self.ffn = FFN(in_channels=channels, hidden_channels=channels*ffn_ratio, act_layer=ffn_act, dropout=ffn_dropout)

        self.drop_path = DropPath(drop_path)
        self.pre_norm = pre_norm

    def forward(self, x, pos_emb=None, mask=None, rpe=None):

        if pos_emb is not None:
            x = x + pos_emb
        shortcut = x
        if self.pre_norm:
            x = self.norm1(x)
        x = self.drop_path(self.attn(x, mask, rpe))

        x = x + shortcut
        if not self.pre_norm:
            x = self.norm1(x)

        shortcut = x
        if self.pre_norm:
            x = self.norm2(x)
        x = self.drop_path(self.ffn(x))
        x = shortcut + x
        if not self.pre_norm:
            x = self.norm2(x)
        return x
