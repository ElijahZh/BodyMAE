import einops
import torch
import torch.nn as nn
from bodymae.layers.mlp import MLPNet
import bodymae.utils.batch as batch_offset


from bodymae.layers.attention import TransformerBlockD, TransformerBlock
from bodymae.geometry.point_cloud import BodyPC
import spconv.pytorch as spconv
import torch_scatter
from torch_geometric.nn import fps, radius
from timm.models.layers import DropPath, trunc_normal_
from collections.abc import Callable
import math
import pointops
from functools import partial


class Embedding(nn.Module):
    """
    Map the channels to bigger size, not position embedding
    """
    def __init__(self, in_channels, embed_channels, norm_layer: Callable = nn.BatchNorm1d, act_layer: Callable = nn.GELU):
        super().__init__()
        self.embed = spconv.SparseSequential(
            spconv.SubMConv3d(
                in_channels,
                embed_channels,
                kernel_size=5,
                padding=1,
                bias=False,
                indice_key="stem"
            )
        )
        if norm_layer is not None:
            self.embed.add(norm_layer(embed_channels))
        if act_layer is not None:
            self.embed.add(act_layer())

    def forward(self, point: BodyPC):

        point.sparse_conv_feat = self.embed(point.sparse_conv_feat)
        point.feat = point.sparse_conv_feat.features
        return point


class PositionEncoding(nn.Module):
    def __init__(self, channels, kernel_size=3, indice_key=None, use_offset=True, use_fourier=True, n_freq=6, use_raw_coord=True):
        super().__init__()

        self.cpe = spconv.SparseSequential(
            spconv.SubMConv3d(channels, channels, kernel_size=kernel_size, bias=True, indice_key=indice_key),
            nn.Linear(channels, channels),
            nn.LayerNorm(channels)
        )

        self.use_offset = use_offset
        self.use_fourier = use_fourier
        hid = max(64, channels // 2)
        if use_offset:
            in_pe = 3 + (0 if not use_fourier else 3*2*n_freq)
            self.alpha_offset = nn.Parameter(torch.tensor(1e-4))
            self.offset_mlp = nn.Sequential(nn.Linear(in_pe, hid),
                                                nn.GELU(),
                                                nn.Linear(hid, 2 * channels if use_fourier else channels)
                                                )

        self.n_freq = n_freq

        self.use_raw_coord = use_raw_coord
        if use_raw_coord:
            self.alpha_raw = nn.Parameter(torch.tensor(1e-3))
            self.raw_coord_mlp = nn.Sequential(nn.Linear(3, hid),
                                               nn.GELU(),
                                               nn.Linear(hid, channels)
                                               )


    def fourier_feats(self, grid_offset):


        freqs = 2.0 ** torch.arange(self.n_freq, device=grid_offset.device) * 2 * math.pi

        x = grid_offset.unsqueeze(-1) * freqs

        sin, cos = torch.sin(x), torch.cos(x)

        return torch.cat((sin, cos), dim=-1).reshape(grid_offset.shape[0], -1)

    def forward(self, point: BodyPC):
        grid_offset = point.grid_offset

        feat = point.feat


        if self.use_offset:
            if self.use_fourier:
                offset_features = torch.cat((grid_offset, self.fourier_feats(grid_offset)), dim=-1)

                offset_pe = self.offset_mlp(offset_features)
                gamma, beta = torch.chunk(offset_pe, 2, dim=-1)

                feat = feat * (1.0 + self.alpha_offset * gamma) + self.alpha_offset * beta

            else:
                offset_pe = self.offset_mlp(grid_offset)
                feat = feat + self.alpha_offset * offset_pe

        if self.use_raw_coord:
            coord_pe = self.raw_coord_mlp(point.coord)
            feat = feat + self.alpha_raw * coord_pe


        sparse_conv_feat = point.sparse_conv_feat
        x = spconv.SparseConvTensor(feat, sparse_conv_feat.indices, sparse_conv_feat.spatial_shape, sparse_conv_feat.batch_size)
        return self.cpe(x).features


class Block(nn.Module):
    def __init__(self, channels, num_heads, patch_size, qkv_bias=True, qk_scale=None, attn_drop=0.0, proj=True, proj_drop=0.0,
                 order_index=0,  ffn_ratio=4, ffn_act=nn.GELU, ffn_dropout=0., drop_path=0., norm_layer: Callable=nn.LayerNorm,
                 kernel_size=3, indice_key=None, use_offset=True, use_fourier=True, n_freq=6, use_raw_coord=True):
        super().__init__()
        self.patch_size = patch_size
        self.order_index = order_index

        self.pe = PositionEncoding(channels, kernel_size=kernel_size, indice_key=indice_key, use_offset=use_offset,
                                   use_fourier=use_fourier, n_freq=n_freq, use_raw_coord=use_raw_coord)

        self.transformer = TransformerBlockD(channels=channels, num_heads=num_heads, qkv_bias=qkv_bias,
                                             qk_scale=qk_scale, attn_drop=attn_drop, use_flash_attn=False, proj=proj,
                                             proj_drop=proj_drop, ffn_ratio=ffn_ratio, ffn_act=ffn_act,
                                             ffn_dropout=ffn_dropout, drop_path=drop_path, norm_layer=norm_layer,
                                             pre_norm=True)

    @torch.no_grad()
    def get_padding_and_inverse(self, point):
        pad_key = "pad"
        unpad_key = "unpad"
        cu_seqlens_key = "cu_seqlens_key"
        if (pad_key not in point.keys() or unpad_key not in point.keys() or cu_seqlens_key not in point.keys()):
            offset = point.offset
            bincount = batch_offset.offset2bincount(offset)
            bincount_pad = (torch.div(bincount + self.patch_size - 1, self.patch_size,
                                      rounding_mode="trunc") * self.patch_size)

            mask_pad = bincount > self.patch_size
            bincount_pad = ~mask_pad * bincount + mask_pad * bincount_pad
            _offset = nn.functional.pad(offset, (1, 0))
            _offset_pad = nn.functional.pad(torch.cumsum(bincount_pad, dim=0), (1, 0))
            pad = torch.arange(_offset_pad[-1], device=offset.device)
            unpad = torch.arange(_offset[-1], device=offset.device)
            cu_seqlens = []
            for i in range(len(offset)):
                unpad[_offset[i]: _offset[i + 1]] += _offset_pad[i] - _offset[i]
                if bincount[i] != bincount_pad[i]:
                    pad[
                    _offset_pad[i + 1] - self.patch_size + (bincount[i] % self.patch_size): _offset_pad[i + 1]] = pad[
                                                                                                                  _offset_pad[
                                                                                                                      i + 1]
                                                                                                                  - 2 * self.patch_size
                                                                                                                  + (
                                                                                                                              bincount[
                                                                                                                                  i] % self.patch_size):
                                                                                                                  _offset_pad[
                                                                                                                      i + 1]
                                                                                                                  - self.patch_size
                                                                                                                  ]
                pad[_offset_pad[i]: _offset_pad[i + 1]] -= _offset_pad[i] - _offset[i]
                cu_seqlens.append(
                    torch.arange(
                        _offset_pad[i],
                        _offset_pad[i + 1],
                        step=self.patch_size,
                        dtype=torch.int32,
                        device=offset.device,
                    )
                )
            point[pad_key] = pad
            point[unpad_key] = unpad
            point[cu_seqlens_key] = nn.functional.pad(
                torch.concat(cu_seqlens), (0, 1), value=_offset_pad[-1]
            )
        return point[pad_key], point[unpad_key], point[cu_seqlens_key]


    def forward(self, point):
        pos_embed = self.pe(point)

        self.patch_size = min(batch_offset.offset2bincount(point.offset).min().tolist(), self.patch_size)

        pad, unpad, cu_seqlens = self.get_padding_and_inverse(point)
        order = point.serialized_order[self.order_index][pad]
        inverse = unpad[point.serialized_inverse[self.order_index]]

        x = self.transformer(x=point.feat[order], pos_emb=pos_embed[order], cu_seqlens=cu_seqlens,
                             patch_size=self.patch_size)

        x = x[inverse]
        point.feat = x
        point.sparse_conv_feat = point.sparse_conv_feat.replace_feature(x)
        return point


class PoolingArea(nn.Module):
    def __init__(self, in_channels, out_channels, fps_radius, ball_radius, grid_size, norm_layer: Callable = nn.BatchNorm1d,
                 act_layer: Callable = nn.GELU, reduce="max", shuffle_orders=True, traceable=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.fps_radius = fps_radius
        self.ball_radius = ball_radius
        self.grid_size = grid_size
        self.traceable = traceable

        assert reduce in ["sum", "mean", "min", "max"]
        self.reduce = reduce
        self.shuffle_orders = shuffle_orders

        self.proj = nn.Sequential(nn.Linear(in_channels + 3, out_channels),
                                  norm_layer(out_channels),
                                  act_layer()
                                  )

    def forward(self, point: BodyPC):
        coord = point.coord
        area = point.area
        offset = point.offset
        batch = point.batch

        new_bincount = (2 * area / (math.sqrt(3) * self.fps_radius * self.fps_radius)).long()

        new_offset = batch_offset.bincount2offset(new_bincount)
        fps_index = pointops.farthest_point_sampling(coord, offset, new_offset)

        new_coord = coord[fps_index]
        new_batch = batch_offset.offset2batch(new_offset)


        row, col = radius(coord, new_coord, self.ball_radius, batch, new_batch, max_num_neighbors=64)
        xyz_normal = coord[col] - new_coord[row]
        new_feat = torch.cat((xyz_normal, point.feat[col]), dim=1)

        new_feat = self.proj(new_feat)
        new_feat = torch_scatter.scatter(new_feat, row, dim=-2, reduce=self.reduce, dim_size=row[-1] + 1)


        new_curvature = point.curvature[fps_index]
        new_normal = point.norm[fps_index]

        new_point = BodyPC(dict(coord=new_coord, feat=new_feat, offset=new_offset, curvature=new_curvature, norm=new_normal,
                                area=point.area, grid_size=self.grid_size))
        if self.traceable:
            new_point["pooling_inverse"] = (col, row)
            new_point["pooling_parent"] = point

        new_point.serialization(shuffle_orders=self.shuffle_orders)
        new_point.sparsify()

        return new_point


class PTTransformer(nn.Module):
    def __init__(self, in_channel=3, embed_channel=32, mlp_list=(64, 128), head_list=(2, 4, 8), enc_depths=(2, 4, 4),
                 patch_list=(256, 256, 256), drop_path=0.2,
                 fps_radius_list=(0.0141, 0.02), ball_radius_list=(0.025, 0.05), grid_size_list=(0.04, 0.08),
                 pe_kernel_size_list=(3, 3, 3), use_offset=True, use_fourier=True, n_freq=6, use_raw_coord=True,
                 traceable=False, reduce="max"):
        super().__init__()
        bn_layer = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        act_layer = nn.GELU

        self.embedding = Embedding(in_channels=in_channel, embed_channels=embed_channel, norm_layer=bn_layer,
                                   act_layer=act_layer)
        enc_drop_path = [x.item() for x in torch.linspace(0, drop_path, sum(enc_depths))]

        self.transformer_layers = nn.ModuleList()
        self.pooling_layers = nn.ModuleList()
        mlp_list = (embed_channel,) + mlp_list
        order = ("z", "z-trans", "hilbert", "hilbert-trans")
        for s in range(len(enc_depths)):
            if s > 0:
                pooling_module = PoolingArea(in_channels=mlp_list[s-1], out_channels=mlp_list[s],
                                             fps_radius=fps_radius_list[s-1], ball_radius=ball_radius_list[s-1],
                                             grid_size=grid_size_list[s-1], norm_layer=bn_layer, act_layer=act_layer, reduce=reduce,
                                             shuffle_orders=True, traceable=traceable
                )
                self.pooling_layers.append(pooling_module)

            enc = nn.Sequential()
            enc_drop_path_ = enc_drop_path[sum(enc_depths[:s]): sum(enc_depths[: s + 1])]
            for i in range(enc_depths[s]):
                block = Block(channels=mlp_list[s], num_heads=head_list[s], patch_size=patch_list[s],
                              qkv_bias=True, qk_scale=None, attn_drop=0.0, proj=True, proj_drop=0.0,
                              order_index=i % len(order),  ffn_ratio=4, ffn_act=act_layer, ffn_dropout=0.,
                              drop_path=enc_drop_path_[i], norm_layer=bn_layer,
                              kernel_size=pe_kernel_size_list[s], indice_key=f"stage{s}-block{i}", use_offset=use_offset,
                              use_fourier=use_fourier, n_freq=n_freq, use_raw_coord=use_raw_coord)
                enc.append(block)
            self.transformer_layers.append(enc)

    def forward(self, point):
        point = self.embedding(point)
        for i in range(len(self.transformer_layers)):
            if i > 0:
                point = self.pooling_layers[i-1](point)

            point = self.transformer_layers[i](point)
        return point


class Group(nn.Module):
    def __init__(self, n_patch=128, k=64):
        super().__init__()
        self.n_patch = n_patch
        self.k = k

    def forward(self, point):
        coord = point.coord
        feat = point.feat
        offset = point.offset
        norm = point.norm
        curvature = point.curvature.unsqueeze(-1)


        new_offset = torch.arange(1, len(offset) + 1, device=offset.device) * self.n_patch
        fps_idx = pointops.farthest_point_sampling(coord, offset, new_offset)
        new_coord = coord[fps_idx]

        neighbors, knn_idx = pointops.knn_query_and_group(feat, coord, offset=offset, new_xyz=new_coord, new_offset=new_offset, nsample=self.k, with_xyz=True)






        neighbors = einops.rearrange(neighbors, "(b n) k c -> b n k c", n=self.n_patch)

        neighbors_coord = neighbors[..., :3]
        neighbors_feat = neighbors[..., 3:]
        center_coord = einops.rearrange(new_coord, "(b n) c -> b n c", n=self.n_patch)

        neighbors_norm = pointops.grouping(knn_idx, norm, coord, with_xyz=False)
        neighbors_curvature = pointops.grouping(knn_idx, curvature, coord, with_xyz=False)

        neighbors_norm = einops.rearrange(neighbors_norm, "(b n) k c -> b n k c", n=self.n_patch)
        neighbors_curvature = einops.rearrange(neighbors_curvature, "(b n) k c -> b n k c", n=self.n_patch)

        return neighbors_coord, neighbors_feat, center_coord, neighbors_norm, neighbors_curvature


class Encoder(nn.Module):
    def __init__(self, mlp_list=(128 + 3, 256, 384)):
        super().__init__()
        self.mlp = MLPNet(mlp_list=mlp_list, active=True, active_type="relu", normalization=True,
                          normalization_type="BatchNormal1d", plain_last=True)

    def forward(self, feat):
        """
        :param feat: (B, N, K, C+3)
        :return: (B, N, D)
        """

        B, N, K, _ = feat.shape
        feat = einops.rearrange(feat, "b n k c -> (b n k) c")
        feat = self.mlp(feat)
        feat = einops.rearrange(feat, "(b n k) c -> (b n) k c", b=B, n=N)
        feat = torch.max(feat, dim=-2)[0]
        feat = einops.rearrange(feat, "(b n) c -> b n c", b=B)
        return feat


class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim=384, depth=6, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., proj=True, drop_path_rate: float | int | list =0.):
        super().__init__()
        self.pos_embed = MLPNet(mlp_list=(3, 128, embed_dim), active_type="gelu", normalization=False, plain_last=True)

        self.blocks = nn.ModuleList()
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        for i in range(depth):
            block = TransformerBlock(channels=embed_dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                     attn_drop=attn_drop_rate, proj=proj, proj_drop=drop_rate,
                                     ffn_ratio=int(mlp_ratio), ffn_act=nn.GELU, ffn_dropout=drop_rate, drop_path=dpr[i],
                                     norm_layer=nn.LayerNorm, pre_norm=True)

            self.blocks.append(block)

    def forward(self, x, xyz):

        pe = self.pos_embed(xyz)
        for block in self.blocks:
            x = block(x, pos_emb=pe)
        return x


class MaskTransformer(nn.Module):
    def __init__(self, mask_ratio=0.7, mlp_list=(128+3, 256, 384), depth=12, drop_path_rate=0.1, num_heads=6):
        super().__init__()
        self.mask_ratio = mask_ratio
        trans_dim = mlp_list[-1]
        self.trans_dim = trans_dim
        self.depth = depth
        self.drop_path_rate = drop_path_rate
        self.num_heads = num_heads

        self.encoder = Encoder(mlp_list=mlp_list)

        self.transformer_encoder = TransformerEncoder(embed_dim=trans_dim, depth=depth, num_heads=num_heads, drop_path_rate=drop_path_rate)
        self.norm = nn.LayerNorm(trans_dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, mean=0., std=.02, a=-2., b=2.)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, mean=0., std=.02, a=-2., b=2.)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _mask_center_rand(self, center, noaug=False):
        '''
            center : B G 3
            --------------
            mask : B G (bool)
        '''
        B, G, _ = center.shape

        if noaug or self.mask_ratio == 0:
            return torch.zeros(center.shape[:2]).bool()

        num_mask = int(self.mask_ratio * G)
        mask = torch.zeros((B, G), device=center.device, dtype=torch.bool)
        mask[:, :num_mask] = True
        row_idx = torch.arange(B).unsqueeze(1).expand(B, G)
        rand_perm = torch.rand(B, G).argsort(dim=1)
        mask = mask[row_idx, rand_perm]
        return mask

    def forward(self, neighborhood, center, neighborhood_features, noaug=False):
        mask = self._mask_center_rand(center, noaug=noaug)
        vis_mask = ~mask
        group_input_tokens = self.encoder(torch.cat((neighborhood, neighborhood_features), dim=-1))

        B, G, C = group_input_tokens.shape



        x_vis = group_input_tokens[vis_mask].reshape(B, -1, C)
        center_vis = center[vis_mask].reshape(B, -1, 3)


        x_vis = self.transformer_encoder(x_vis, center_vis)

        x_vis = self.norm(x_vis)
        return x_vis, mask


class TransformerDecoder(nn.Module):
    def __init__(self, embed_dim=384, depth=4, num_heads=6, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., proj=True, drop_path_rate: float | int | list = 0.1, norm_layer=nn.LayerNorm):
        super().__init__()
        self.pos_embed = MLPNet(mlp_list=(3, 128, embed_dim), active_type="gelu", normalization=False, plain_last=True)
        self.blocks = nn.ModuleList()
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        for i in range(depth):
            block = TransformerBlock(channels=embed_dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                     attn_drop=attn_drop_rate, proj=proj, proj_drop=drop_rate,
                                     ffn_ratio=int(mlp_ratio), ffn_act=nn.GELU, ffn_dropout=drop_rate, drop_path=dpr[i],
                                     norm_layer=nn.LayerNorm, pre_norm=True)

            self.blocks.append(block)

        self.norm = norm_layer(embed_dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, xyz, return_token_num):
        pe = self.pos_embed(xyz)
        for _, block in enumerate(self.blocks):
            x = block(x, pe)

        x = self.norm(x[:, -return_token_num:])
        return x

class BodyMAE(nn.Module):
    def __init__(self, in_channel=3, embed_channel=64, mlp_list=(128, 256), head_list=(2, 4, 8), enc_depths=(2, 2, 2),
                 patch_list=(256, 256, 256), drop_path=0.2,
                 fps_radius_list=(0.0141, 0.02), ball_radius_list=(0.04, 0.08), grid_size_list=(0.02, 0.04),
                 pe_kernel_size_list=(3, 3, 3), use_offset=True, use_fourier=False, n_freq=6, use_raw_coord=True,
                 traceable=False, reduce="max",
                 n_patch=128, k=64,
                 mask_ratio=0.7, enc_mlp_list=(512, 384), enc_depth=12, enc_drop_path_rate=0.1, enc_num_heads=6,
                 dec_depth=4, dec_num_heads=6, dec_drop_path_rate=0.1,

                 use_extra=False,
                 ):
        super().__init__()
        self.pt_transformer = PTTransformer(in_channel=in_channel, embed_channel=embed_channel, mlp_list=mlp_list,
                                            head_list=head_list, enc_depths=enc_depths, patch_list=patch_list,
                                            drop_path=drop_path, fps_radius_list=fps_radius_list,
                                            ball_radius_list=ball_radius_list, grid_size_list=grid_size_list,
                                            pe_kernel_size_list=pe_kernel_size_list, use_offset=use_offset,
                                            use_fourier=use_fourier, n_freq=n_freq, use_raw_coord=use_raw_coord,
                                            traceable=traceable, reduce=reduce)
        self.grouping = Group(n_patch=n_patch, k=k)
        enc_mlp_list = (mlp_list[-1] + 3,) + enc_mlp_list
        self.MAE_encoder = MaskTransformer(mask_ratio=mask_ratio, mlp_list=enc_mlp_list, depth=enc_depth,
                                           drop_path_rate=enc_drop_path_rate, num_heads=enc_num_heads)
        trans_dim = enc_mlp_list[-1]
        self.trans_dim = trans_dim
        self.mask_token = nn.Parameter(torch.zeros(1, 1, trans_dim))
        self.MAE_decoder = TransformerDecoder(embed_dim=trans_dim, depth=dec_depth, num_heads=dec_num_heads,
                                              drop_path_rate=dec_drop_path_rate)


        self.coord_head = nn.Sequential(
            nn.Conv1d(trans_dim, 3*k, 1)
        )

        self.use_extra = use_extra
        if use_extra:
            self.normal_head = nn.Sequential(
                nn.Conv1d(trans_dim, 3*k, 1)
            )
            self.curvature_head = nn.Sequential(
                nn.Conv1d(trans_dim, 1*k, 1)
            )

        trunc_normal_(self.mask_token, std=.02)

    def forward(self, data_dict):
        point = BodyPC(data_dict)
        point.serialization()
        point.sparsify()

        point = self.pt_transformer(point)






        neighbors_coord, neighbors_feat, center_coord, neighbors_normal, neighbors_curvature = self.grouping(point)


        x_vis, mask = self.MAE_encoder(neighbors_coord, center_coord, neighbors_feat)
        B, N_vis, C = x_vis.shape


        xyz_vis = center_coord[~mask].reshape(B, -1, 3)
        xyz_mask = center_coord[mask].reshape(B, -1, 3)
        N_mask = xyz_mask.shape[1]
        xyz_full = torch.cat((xyz_vis, xyz_mask), dim=1)

        x_mask = self.mask_token.expand(B, N_mask, -1)
        x_full = torch.cat((x_vis, x_mask), dim=1)

        x_rec = self.MAE_decoder(x_full, xyz_full, N_mask)
        x_rec = einops.rearrange(x_rec, "b n c -> b c n")

        x_rec_coord_offset = self.coord_head(x_rec)
        x_rec_coord_offset = einops.rearrange(x_rec_coord_offset, "b (c k) n -> (b n) k c", c=3)
        gt_coord_offset = neighbors_coord[mask]

        gt_normal = neighbors_normal[mask]
        gt_curvature = neighbors_curvature[mask]

        if self.use_extra:
            x_rec_normal = self.normal_head(x_rec)
            x_rec_normal = einops.rearrange(x_rec_normal, "b (c k) n -> (b n) k c", c=3)
            x_rec_curvature = self.curvature_head(x_rec)
            x_rec_curvature = einops.rearrange(x_rec_curvature, "b (c k) n -> (b n) k c", c=1)
            return x_rec_coord_offset, gt_coord_offset, x_rec_normal, gt_normal, x_rec_curvature, gt_curvature

        return x_rec_coord_offset, gt_coord_offset


