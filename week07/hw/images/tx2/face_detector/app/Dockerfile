FROM tensorflow

RUN mkdir -p /app
WORKDIR /app
COPY . /app

RUN apt update

ENV DEBIAN_FRONTEND=noninteractive

RUN apt update \
  && apt install -y wget python3-pip python3-opencv vim-tiny
RUN pip3 install paho-mqtt
RUN pip3 install mtcnn