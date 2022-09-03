FROM python:3.10
ARG HOST_PERMS=1000:1000
COPY requirements.txt /mnt
RUN chown $HOST_PERMS /mnt/requirements.txt
RUN pip install --no-cache-dir -r /mnt/requirements.txt 
