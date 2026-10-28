"""Build the backend for bodymae.losses.chamfer."""
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="bodymae-chamfer-distance",
    version="0.1.0",
    packages=[],
    ext_modules=[CUDAExtension(
        name="bodymae_chamfer_cuda",
        sources=["chamfer_cuda.cpp", "chamfer.cu"],
    )],
    cmdclass={"build_ext": BuildExtension},
)
