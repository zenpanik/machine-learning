Using 22.07 pythorch container doesnt really work for llm fine tunning.<br>
The reason is that even if the host machine has cuda 12.2 the cuda in the container may differ.<br>
For unsloth we need cuda >= 11.8. The 22.08 container is with 11.7 !<br>

Instructions how to get API key for nvidia container service:

https://build.nvidia.com/meta/llama-3_1-405b-instruct?snippet_tab=Docker

Note: containers are really big and download is slow ... 

Downloading 23.07 <br>
https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-24-01.html <br>
according to the docs it should be with cuda 12.1.1<br>

Then we need to install unsloth for pythorch 23 and cuda 12.1:
```
pip install "unsloth[cu121-torch230] @ git+https://github.com/unslothai/unsloth.git"
```

