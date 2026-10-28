import torch


def offset2batch(offset):
    bincount = offset2bincount(offset)

    batch = torch.arange(len(bincount), device=offset.device, dtype=torch.long).repeat_interleave(bincount)
    return batch


def batch2offset(batch):
    return torch.cumsum(batch.bincount(), dim=0).long()


def batch2offset_zero(batch):
    _offset = torch.nn.functional.pad(torch.cumsum(batch.bincount(), dim=0), (1, 0)).long()


    return _offset

def offset2bincount(offset):
    return torch.diff(offset, prepend=torch.tensor([0], device=offset.device, dtype=torch.long))

def batch2bincount(batch):
    return offset2bincount(batch2offset(batch))


def bincount2offset(bincount):
    return torch.cumsum(bincount, dim=0).long()

def offset_zero(offset):
    _offset = torch.nn.functional.pad(offset, (1, 0)).long()
    return _offset
