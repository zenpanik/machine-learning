FROM tensorflow/tensorflow:devel-gpu AS tf_build
LABEL authors="zenpanik"
RUN chown $HOST_PERMS /mnt/tensorflow-2.8.0-cp38-cp38-linux_x86_64.whl
RUN pip uninstall tensorflow  # remove current version

RUN pip install /mnt/tensorflow-2.8.0-cp38-cp38-linux_x86_64.whl
RUN cd /tmp  # don't import from source directory
RUN python -c "import tensorflow as tf; print(\"Num GPUs Available: \", len(tf.config.list_physical_devices('GPU')))"

FROM tf_build AS jupyter
ARG NB_USER="zenpanik"
ARG NB_UID="1000"
ARG NB_GID="100"
USER root
RUN pip install --no-cache-dir jupyterlab
RUN pip install --no-cache-dir jupyterlab-drawio
RUN pip install --no-cache-dir ipyleaflet "plotly>=4.14.3" "ipywidgets>=7.5"

EXPOSE 8888

# Configure container startup
USER ${NB_UID}
CMD jupyter lab


