FROM nvcr.io/nvidia/cuda:12.4.1-cudnn-devel-ubuntu20.04

RUN apt-get update && apt-get install -y git

RUN git clone https://github.com/NVIDIA/cutlass.git