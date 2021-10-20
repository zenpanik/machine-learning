# 1. Playground [Local Experimentation Environment]

For Data Collection and ML experiments I am using one of my old mining workstation. 

Workstation specification:

Dell T7500
CPU     :   Intel® Xeon(R) CPU X5650 @ 2.67GHz × 6 
Memory  :   23.5 GB
GPU     :   2x GTX 1080TI OC 11G
Disk    :   256 GB SSD & 1TB HDD for storage
OS      :   Ubuntu 18.04.6 LTS

The Playground contains several services:
1. Jupyter Lab with GPU support
2. MLFlow
3. Postgres DB
4. MinIO
5. Airflow
6. Neo4J Graph DB (Community edition)

For Production I may want to replace docker engine with kubernetes so in 2. I will write several paragraphs on how to build production ready containers.

## 1.1. Data Science 

### 1.1.1. Build Tensorflow from Source 
The CPU of this machine is old and Tensorflow is not working out of the box (TF > 1.15 with GPU support) - missing AVX instructions. That is why we need to build TF from source.

After couple unsuccessful trials to build TF on the host machine and move it to container, I decided to build it NVIDIA development container.

The only requirement on the Host machine is that there need to be up-to-date NVIDIA driver. Currently latest NVIDIA driver is:
```
nvidia-smi
```
```
Sun Oct  3 21:19:53 2021       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 470.57.02    Driver Version: 470.57.02    CUDA Version: 11.4     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  NVIDIA GeForce ...  On   | 00000000:03:00.0  On |                  N/A |
|  0%   50C    P5    20W / 140W |    737MiB / 11177MiB |     10%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
|   1  NVIDIA GeForce ...  On   | 00000000:04:00.0 Off |                  N/A |
|  0%   55C    P2    97W / 140W |   3247MiB / 11178MiB |    100%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|    0   N/A  N/A      3280      G   /usr/lib/xorg/Xorg                 24MiB |
|    0   N/A  N/A      3325      G   /usr/bin/gnome-shell               82MiB |
|    0   N/A  N/A      3668      G   /usr/lib/xorg/Xorg                339MiB |
|    0   N/A  N/A      3810      G   /usr/bin/gnome-shell               45MiB |
|    0   N/A  N/A      4153      G   /usr/lib/firefox/firefox          134MiB |
|    0   N/A  N/A     26122      G   ...AAAAAAAAA= --shared-files       51MiB |
|    0   N/A  N/A     37447      G   /usr/lib/firefox/firefox            2MiB |
|    0   N/A  N/A     37650      G   /usr/lib/firefox/firefox            2MiB |
|    0   N/A  N/A     37938      G   ...AAAAAAAAA= --shared-files       44MiB |
|    1   N/A  N/A      1963      C   /home/dell/miner/trex/t-rex      3243MiB |
+-----------------------------------------------------------------------------+
```

Following official steps from Tensorflow documentation:

Currently the latest version of tensorflow is 2.8 and the selected python version is 3.8

https://www.tensorflow.org/install/source

Pull container image from docker hub and clone tensorflow repository:
```
docker pull tensorflow/tensorflow:devel-gpu
docker run --gpus all -it -w /tensorflow -v $PWD:/mnt -e HOST_PERMS="$(id -u):$(id -g)" \
    tensorflow/tensorflow:devel-gpu bash
git clone https://github.com/tensorflow/tensorflow.git  # within the container, download the latest source code & change version if needed
```

The next step is to configure the build:
```
./configure  # answer prompts or use defaults
```

Build the package:
```
bazel build --config=opt --config=cuda //tensorflow/tools/pip_package:build_pip_package
```

Create the python wheel:
```
./bazel-bin/tensorflow/tools/pip_package/build_pip_package /mnt  # create package
```

Change permissions of the whl:
```
chown $HOST_PERMS /mnt/tensorflow-[VERSION-TAGS].whl
```

Test the package:
```
pip uninstall tensorflow  # remove current version

pip install /mnt/tensorflow-version-tags.whl
cd /tmp  # don't import from source directory
python -c "import tensorflow as tf; print(\"Num GPUs Available: \", len(tf.config.list_physical_devices('GPU')))"
```

*Name of the whl file:*
```
tensorflow-2.8.0-cp38-cp38-linux_x86_64.whl
```


### 1.1.2. Build DS Jupyter Container

--> Use the tensorflow docker container for base  
--> Build a container with tensorflow from 1.1.1.  
--> Jupyter lab   
--> MLFlow python libraries  
--> Neo4J python  libraries  
--> Boto3  
--> Plotly  

<span style="color:red"> Need to mount volume `$PWD:/mnt` where the tensorflow whl is located</span>.

You can find the complete Dockerfile in `infrastructure/data-science/Dockerfile`  

Use the latest gpu enabled tensorflow
```
FROM tensorflow/tensorflow:devel-gpu AS tf_build # 
LABEL authors="zenpanik"
```

Uninstall tensorflow (if installed) and install the package built in previous step. Then test it
```
RUN chown $HOST_PERMS /mnt/tensorflow-2.8.0-cp38-cp38-linux_x86_64.whl
RUN pip uninstall tensorflow  # remove current version

RUN pip install /mnt/tensorflow-2.8.0-cp38-cp38-linux_x86_64.whl
RUN cd /tmp  # don't import from source directory
RUN python -c "import tensorflow as tf; print(\"Num GPUs Available: \", len(tf.config.list_physical_devices('GPU')))"
```

Next install packages from requirements.txt
```
FROM tf_build AS jupyter
USER root
RUN pip install --no-cache-dir -r /mnt/requirements.txt 

# Configure container startup
CMD jupyter lab
```

After we have specified all the steps required we need to build the image `using docker` build command:
```
cd infrastructure/data-science
docker build --volume $PWD:/mnt -e HOST_PERMS="$(id -u):$(id -g)" .
```

## 1.2. Build MLflow Server Container

This one is very simple:  
Use python 3.8 as base:  
```
FROM python:3.8
ARG HOST_PERMS=1000:1000
```
Then install `requirements.txt` using pip:  
```
COPY requirements.txt /mnt
RUN chown $HOST_PERMS /mnt/requirements.txt
RUN pip install --no-cache-dir -r /mnt/requirements.txt 
```

## 1.3. Postgres Database and PG admin

Here I am using the bitnami postgresql image:
```
bitnami/postgresql:latest
```

And 
```
dpage/pgadmin4
```

See docker-compose for more details about this deployment

## 1.4. minio (Local S3 storage)
I am using the official latest minio image:

```
minio/minio:latest
```

## 1.5. Neo4J graph database
official community edition image:
```
neo4j:4.2.5-community
```


## All services
In order to make all of the services above work and have network access to each other I am using docker-compose. All services are in the same network - `postgres`.  
I will point out only some of the important environment variables here:

### AWS access keys
```
AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY
```

To obtain the correct access key id and secret key you need to deploy the whole stack once. Then go in the minio UI and use the MINIO_ROOT_PASSWORD & MINIO_ROOT_USER to login. After that you need to create a bucket and pair of key id and secret key. You can then update the .env file with these values.

### Volumes
You need to specify folders where to store:
- postgres data
- minio (S3) data
- datasets and jupyter notebooks (Data Science data)
- Neo4J data




*Notes*:

clean docker environment (PRUNE WILL DELETE ALL IMAGES):
```
docker system prune -a
```

# 2. Scheduler [Local Automation Environment]


