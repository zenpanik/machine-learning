Using 22.07 pythorch container doesnt really work for llm fine tunning.<br>
The reason is that even if the host machine has cuda 12.2 the cuda in the container may differ.<br>
For unsloth we need cuda >= 11.8. The 22.08 container is with 11.7 !<br>

Instructions how to get API key for nvidia container service:

https://build.nvidia.com/meta/llama-3_1-405b-instruct?snippet_tab=Docker

Note: containers are really big and download is slow ... 

Downloading 23.07 <br>
https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-24-01.html <br>
according to the docs it should be with cuda 12.1.1<br>

Then we need to install unsloth:
- first check nvidia-smi
```
nvidia-smi
```
```
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.183.01             Driver Version: 535.183.01   CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA GeForce GTX 1080        Off | 00000000:04:00.0 Off |                  N/A |
|  0%   21C    P8               9W / 210W |     15MiB /  8192MiB |      0%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+
                                                                                         
+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
+---------------------------------------------------------------------------------------+
```

- Then check cuda compiler driver version
```
nvcc --version
```
```
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2023 NVIDIA Corporation
Built on Mon_Apr__3_17:16:06_PDT_2023
Cuda compilation tools, release 12.1, V12.1.105
Build cuda_12.1.r12.1/compiler.32688072_0
```

- Install latest setuptools
```
python -m pip install --upgrade pip wheel setuptools
```

- Install unsloth zoo (optional)
```
pip install unsloth_zoo
```

- Finally install unsloth
```
pip install "unsloth[cu121-torch240] @ git+https://github.com/unslothai/unsloth.git"
```


