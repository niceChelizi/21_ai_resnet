from easydict import EasyDict as edict
import os
import numpy as np
import matplotlib.pyplot as plt
import mindspore
import mindspore.dataset as ds
from mindspore.dataset.vision import c_transforms as vision
from mindspore import context
import mindspore.nn as nn
from mindspore.train import Model
from mindspore.nn.optim.momentum import Momentum
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor
from mindspore import Tensor
from mindspore.train.serialization import export
from mindspore.train.loss_scale_manager import FixedLossScaleManager
from mindspore.train.serialization import load_checkpoint, load_param_into_net
import mindspore.ops as ops
from tensorbay import GAS
from tensorbay.dataset import Dataset



gas = GAS("Accesskey-7b4ca5d2a6e9ea5f3273c8506050e7ed")
dataset = Dataset("RP2K", gas)
os.mkdir("test")
os.mkdir("work")
os.mkdir("data")
os.chdir("test")#
segment = dataset["test"]
a = segment[0]
Li = segment[0:2500]
for i in Li:
    name = i.path
    fp = i.open()
    a = fp.read()
    classification_category = i.label.classification.category
    if os.path.exists(classification_category):
        with open(classification_category+"/"+name,"wb") as f:
            f.write(a)
    else:
        os.mkdir(classification_category)
        with open(classification_category+"/"+name, "wb") as f:
            f.write(a)

count = 0
dic = {}
for i in os.listdir():
    dic[i] = count
    count += 1

os.chdir("data")#
segment = dataset["train"]
a = segment[0]
Li = segment[0:100]
for i in Li:
    name = i.path
    fp = i.open()
    a = fp.read()
    classification_category = i.label.classification.category
    if os.path.exists(classification_category):
        with open(classification_category+"/"+name,"wb") as f:
            f.write(a)
    else:
        os.mkdir(classification_category)
        with open(classification_category+"/"+name, "wb") as f:
            f.write(a)
 
    
cfg = edict({
    'data_path': 'data',  # 训练数据集，如果是zip文件需要解压
    'test_path': 'test',  # 测试数据集，如果是zip文件需要解压
    'HEIGHT': 224,  # 图片高度
    'WIDTH': 224,  # 图片宽度
    '_R_MEAN': 123.68,
    '_G_MEAN': 116.78,
    '_B_MEAN': 103.94,
    '_R_STD': 1,
    '_G_STD': 1,
    '_B_STD': 1,
    '_RESIZE_SIDE_MIN': 256,
    '_RESIZE_SIDE_MAX': 512,

    'batch_size': 32,
    'num_class': 1167,  # 分类类别
    'epoch_size': 200,  # 训练次数
    'loss_scale_num': 250,

    'prefix': 'resnet-ai',
    'directory': './model_resnet',
    'save_checkpoint_steps': 10,
})
os.chdir("../work")
def read_data(path, config, usage="train"):
    # 从目录中读取图像的源数据集。
    dataset = ds.ImageFolderDataset(path,
                                    class_indexing=dic)
    # define map operations
    decode_op = vision.Decode()
    normalize_op = vision.Normalize(mean=[cfg._R_MEAN, cfg._G_MEAN, cfg._B_MEAN],
                                    std=[cfg._R_STD, cfg._G_STD, cfg._B_STD])
    resize_op = vision.Resize(cfg._RESIZE_SIDE_MIN)
    center_crop_op = vision.CenterCrop((cfg.HEIGHT, cfg.WIDTH))
    horizontal_flip_op = vision.RandomHorizontalFlip()
    channelswap_op = vision.HWC2CHW()
    random_crop_decode_resize_op = vision.RandomCropDecodeResize((cfg.HEIGHT, cfg.WIDTH), (0.5, 1.0), (1.0, 1.0),
                                                                 max_attempts=100)
    if usage == 'train':
        dataset = dataset.map(input_columns="image", operations=random_crop_decode_resize_op)
        dataset = dataset.map(input_columns="image", operations=horizontal_flip_op)
    else:
        dataset = dataset.map(input_columns="image", operations=decode_op)
        dataset = dataset.map(input_columns="image", operations=resize_op)
        dataset = dataset.map(input_columns="image", operations=center_crop_op)

    dataset = dataset.map(input_columns="image", operations=normalize_op)
    dataset = dataset.map(input_columns="image", operations=channelswap_op)

    if usage == 'train':
        dataset = dataset.shuffle(buffer_size=10000)  # 10000 as in imageNet train script
        dataset = dataset.batch(cfg.batch_size, drop_remainder=True)
    else:
        dataset = dataset.batch(1, drop_remainder=True)
    dataset = dataset.repeat(1)
    dataset.map_model = 4
    return dataset

de_train = read_data(cfg.data_path, cfg, usage="train")
de_test = read_data(cfg.test_path, cfg, usage="test")
print('训练数据集数量：', de_train.get_dataset_size() * cfg.batch_size)
print('测试数据集数量：', de_test.get_dataset_size())
de_dataset = de_train
data_next = de_dataset.create_dict_iterator(output_numpy=True).__next__()
