#!/bin/bash

apt update
apt install python3.10 python3.10-distutils python3.10-dev cuda-toolkit-10-2
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python3.10 get-pip.py
pip3.10 install -r requirements.txt
pip3.10 install dvc mlflow

tar -xvf ../cudnn.tar.gz
cp cudnn-*-archive/include/cudnn*.h /usr/local/cuda/include 
cp -P cudnn-*-archive/lib/libcudnn* /usr/local/cuda/lib64 
chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda:/usr/local/cuda/include:/usr/local/cuda/lib64/
dvc remote modify origin --local auth basic
dvc remote modify origin --local user $1 
dvc remote modify origin --local password $2
dvc pull -r origin
dvc pull -r origin # second pass to download missed files 
ldconfig
