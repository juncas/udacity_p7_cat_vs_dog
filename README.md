# Udacity 机器学习进阶P7:猫狗大战

 ## 简介
这是Udacity 机器学习（进阶）毕业项目猫狗大战的全部实现过程。供后续同学参考。想要了解每个文件的内容，请先阅读report.pdf文件，该文件为毕业项目的详细报告。本文的实现参考了[培神的实现](https://github.com/ypwhs/dogs_vs_cats)。如果参考了本实现或者培神的方法，请记得引用。

## 使用到的库
- os
- shutil
- numpy
- random
- tqdm
- time
- PIL
- h5py
- pandas
- keras
- sklearn

## 训练所用机器
- CPU：i7-8700k
- 内存： 16G DDR4 2400
- GPU: 0-GTX1070Ti 1-Tesla K40c
- 其他不重要

## 操作系统
Ubuntu 16.04 64位

## 特征提取(K40c)
- ResNet50: 1100s
- Xception: 1350s
- DenseNet169: 2200s
- Inception_V3: 1500s
- NASNetMobile: 1260s

## 训练时间
- DenseNet169(on GTX1070Ti)： 600s/Epoch*20 Epochs 
- Inception_V3(on K40c): 1s/Epoch*20 Epochs
- Mergenet1(on K40c): 1s/Epoch*20 Epochs
- Mergenet2(on K40c): 1s/Epoch*20 Epochs 
- Mergenet3(on K40c): 1s/Epoch*20 Epochs 
- ResNet50: 
	- 119s/Epoch*10 (Freeze) 
	- 124s/Epoch*5 (Fine-tuning 1)
	- 132s/Epoch*5 (Fine-tuning 2)
	- 695s/Epoch*10 (Fine-tuning 3)
- Xception(on GTX1070Ti): 
	- 119s/Epoch*10 (Freeze) 
	- 124s/Epoch*5 (Fine-tuning 1)
	- 132s/Epoch*5 (Fine-tuning 2)
	- 695s/Epoch*10 (Fine-tuning 3)
