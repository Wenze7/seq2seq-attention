Requirements:
Python3
PyTorch >= 0.4
gensim
tensorboardX
cytoolz


Quickstart:
[1] Decompress the file "dataset.zip" into current folder.

[2] Training with GPUs:
CUDA_VISIBLE_DEVICES=0 python main.py --data=./dataset --path=./log

[3] Training without GPUs:
python main.py --data=./dataset --path=./log --no-cuda
