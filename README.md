# BodyMAE architecture

This repository provides only the BodyMAE model implementation, including the
architecture, reconstruction losses, and model configuration. The study data
cannot be shared due to participant privacy restrictions.

## Installation

Install a CUDA-enabled PyTorch build compatible with your CUDA environment
before continuing. Select compatible builds of spconv, torch-scatter,
torch-cluster, and Flash Attention for that PyTorch/CUDA combination, following
their installation instructions. The attention constructors require
`flash_attn` to be importable with their current default flags.
`requirements.txt` lists the required packages; it is a dependency inventory,
not a tested lockfile, and some packages may need platform-specific wheels or
source builds.

Then install the remaining dependencies, the pinned pointops package, the
bundled Chamfer CUDA extension, and BodyMAE:

```sh
python -m pip install --no-build-isolation -r requirements.txt
python -m pip install --no-build-isolation -r requirements-pointops.txt
python -m pip install --no-build-isolation ./native/chamfer_distance
```

## Input dictionary (`data_dict`)

| Key         | Type and shape            | Meaning                                                                                                               |
| ----------- | ------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| `coord`     | `torch.float32`, `[N, 3]` | XYZ point coordinates in meters.                                                                                      |
| `norm`      | `torch.float32`, `[N, 3]` | Unit surface normals, aligned with the coordinates.                                                                   |
| `feat`      | `torch.float32`, `[N, 3]` | Input point features; normally a copy of `norm`.                                                                      |
| `area`      | `torch.float32`, `[B]`    | Surface area of each input sample in square meters, in batch order. Used to determine area-dependent sampling counts. |
| `offset`    | `torch.long`, `[B]`       | Cumulative point counts: `[N_1, N_1 + N_2, ..., N]`. These are end positions, not starting indices.                   |
| `grid_size` | Positive Python `float`   | Initial voxel size in meters; the supplied configuration uses `0.01`.                                                 |
