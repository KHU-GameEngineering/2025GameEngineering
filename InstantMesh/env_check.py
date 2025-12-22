import torch
import numpy as np

print("=== Torch / CUDA ===")
print("torch version:", torch.__version__)
print("torch cuda version:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())

try:
    import nvdiffrast.torch as dr
    print("\n=== nvdiffrast ===")
    print("nvdiffrast module:", dr.__file__)
    ctx = dr.RasterizeCudaContext()
    print("nvdiffrast RasterizeCudaContext(): OK")
except Exception as e:
    print("nvdiffrast ERROR:", repr(e))

try:
    import xformers
    print("\n=== xformers ===")
    
    print("xformers version:", xformers.__version__)
except Exception as e:
    print("xformers ERROR:", repr(e))

print("\n=== NumPy ===")
print("numpy version:", np.__version__)
