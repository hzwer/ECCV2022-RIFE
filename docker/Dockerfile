FROM python:3.8-slim

# install deps
RUN apt-get update && apt-get -y install \
    bash ffmpeg

# setup RIFE
WORKDIR /rife
COPY . .
RUN pip3 install -r requirements.txt

ADD docker/rife.sh /usr/local/bin/rife
RUN chmod +x /usr/local/bin/rife

# add pre-trained models
COPY docker/pretrained_models /rife/train_log

WORKDIR /host
ENTRYPOINT ["rife"]

ENV NVIDIA_DRIVER_CAPABILITIES all