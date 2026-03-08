import os
from dotenv import load_dotenv
load_dotenv()

nvidia_base = os.getenv("NVIDIA_BASE")
pkgs = [
    "cublas\\bin",
    "cuda_nvrtc\\bin", 
    "cudnn\\bin",
    "cuda_runtime\\bin",
    "cuda_cupti\\bin",
    "cufft\\bin",
    "nvjitlink\\bin",
]
for pkg in pkgs:
    path = os.path.join(nvidia_base, pkg)
    if os.path.exists(path):
        os.add_dll_directory(path)
        print(f"Added: {path}")

import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))