FROM python:3.7.16

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install opencv-python
RUN pip install numpy
RUN pip install torch
RUN pip install torchvision
RUN pip install opencv-contrib-python
WORKDIR /app