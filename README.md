docker run -it -v /dev/shm:/dev/shm --runtime=nvidia swr.cn-south-1.myhuaweicloud.com/mindspore/mindspore-gpu-cuda10.1:1.5.0 /bin/bash
pip install tensorbay
pip install --upgrade pip

sudo apt-get install git 
git clone  https://github.com/niceChelizi/21_ai_resnet.git

pip install matplotlib
pip install easydict 
cd copy1
python dataPre.py
python resNet50.py
