import torch
import spconv.pytorch as spconv
from addict import Dict
import bodymae.utils.batch as batch_offset
import torch.nn.functional as F
from bodymae.geometry.serialization import encode


class BodyPC(Dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if "batch" not in self.keys() and "offset" in self.keys():
            self["batch"] = batch_offset.offset2batch(self.offset)
        elif "offset" not in self.keys() and "batch" in self.keys():
            self["offset"] = batch_offset.batch2offset(self.batch)


        if "grid_coord" not in self.keys():

            assert "grid_size" in self.keys()
            positive_coord = self.coord - self.coord.min(0)[0]
            self.grid_coord = torch.div(positive_coord, self.grid_size, rounding_mode="trunc").int()
            voxel_center_coord = (self.grid_coord.float() + 0.5) * self.grid_size

            self.grid_offset = (positive_coord - voxel_center_coord) / self.grid_size
            self.grid_offset = torch.clamp(self.grid_offset, -0.5 + 1e-6, 0.5 - 1e-6)

    def normalization_feat(self):
        if "feat" in self.keys():
            self.feat = F.normalize(self.feat, p=2, dim=1, eps=1e-12)

    def serialization(self, order=("z", "z-trans", "hilbert", "hilbert-trans"), depth=None, shuffle_orders=True):
        assert "batch" in self.keys()
        if depth is None:
            depth = int(self.grid_coord.max()).bit_length()
        self["serialized_depth"] = depth

        assert depth * 3 + len(self.offset).bit_length() <= 63
        assert depth <= 16
        code = [encode(self.grid_coord, self.batch, depth, order=order_) for order_ in order]
        code = torch.stack(code)
        order = torch.argsort(code)
        inverse = torch.zeros_like(order).scatter_(dim=1, index=order, src=torch.arange(0, code.shape[1], device=order.device).repeat(code.shape[0], 1))



        if shuffle_orders:
            perm = torch.randperm(code.shape[0])
            code = code[perm]
            order = order[perm]
            inverse = inverse[perm]

        self["serialized_code"] = code
        self["serialized_order"] = order
        self["serialized_inverse"] = inverse

    def sparsify(self, pad=96):
        assert {"coord", "feat", "batch", "grid_coord"}.issubset(self.keys())

        if "sparse_shape" in self.keys():
            sparse_shape = self.sparse_shape
        else:
            sparse_shape = torch.add(torch.max(self.grid_coord, dim=0).values, pad).tolist()

        sparse_conv_feat = spconv.SparseConvTensor(
            features=self.feat,
            indices=torch.cat([self.batch.unsqueeze(-1).int(), self.grid_coord.int()], dim=1).contiguous(),
            spatial_shape=sparse_shape,
            batch_size=self.batch[-1].tolist() + 1,
        )

        self["sparse_shape"] = sparse_shape
        self["sparse_conv_feat"] = sparse_conv_feat
