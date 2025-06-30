#FROM nvcr.io/nvidia/pytorch:23.06-py3
FROM nvcr.io/nvidia/pytorch:23.06-py3

RUN apt update && apt install -y ffmpeg sox git wget unzip \
 && pip install --upgrade pip \
 && pip install nemo_toolkit[all] torchaudio librosa \
 && pip install wyoming

WORKDIR /workspace

