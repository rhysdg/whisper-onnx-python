#!/bin/bash

sudo add-apt-repository -y ppa:ubuntu-toolchain-r/test
sudo apt update
sudo apt install -y g++-11

sudo apt-get install python3-pip
sudo apt-get install libopenblas-base libopenmpi-dev libomp-dev portaudio19-dev espeak ffmpeg libespeak1 python3-pyaudio

###grabbing all relevant wheels
wget https://nvidia.box.com/shared/static/g74cjqh8fcyd9faobm7yeif8mmxdvc0g.whl -O onnxruntime_gpu-1.16.0-cp311-cp311-linux_aarch64.whl

pip install -r requirements.txt

echo "done!"