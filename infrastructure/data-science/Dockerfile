FROM tensorflow/tensorflow:devel-gpu AS tf_build
LABEL authors="zenpanik"
ARG HOST_PERMS=1000:1000
RUN python --version
COPY tensorflow-2.8.0-cp38-cp38-linux_x86_64.whl /mnt
COPY requirements.txt /mnt
RUN chown $HOST_PERMS /mnt/tensorflow-2.8.0-cp38-cp38-linux_x86_64.whl
RUN chown $HOST_PERMS /mnt/requirements.txt
RUN pip uninstall -y enum34
RUN pip uninstall tensorflow  # remove current version

RUN pip install /mnt/tensorflow-2.8.0-cp38-cp38-linux_x86_64.whl
RUN cd /tmp  # don't import from source directory
RUN python -c "import tensorflow as tf; print(\"Num GPUs Available: \", len(tf.config.list_physical_devices('GPU')))"

FROM tf_build AS jupyter
USER root
RUN pip install --no-cache-dir -r /mnt/requirements.txt 

# Configure container startup
CMD jupyter lab


