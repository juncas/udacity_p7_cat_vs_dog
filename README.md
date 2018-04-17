# Udacity 机器学习进阶P7:猫狗大战
![cat.vs.dog](https://github.com/juncas/udacity_p7_cat_vs_dog/blob/master/images/dog_cat.jpg)

## 简介
这是Udacity 机器学习（进阶）毕业项目猫狗大战的全部实现过程。供后续同学参考。

![最优模型才随机测试样本上的预测](https://github.com/juncas/udacity_p7_cat_vs_dog/blob/master/images/pred_samples.png)

本文项目采用**融合+迁移+调参**的方法实现，最终的模型Mergenet3获得的得分为0.03546。这个分数可以排到当时[Kaggle天梯](https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/leaderboard)的前0.5%，第六名。实现参考了[培神的实现](https://github.com/ypwhs/dogs_vs_cats)。如果参考了本实现或者培神的方法，请记得引用。

想要了解每个文件的内容，请先阅读[report.pdf](https://github.com/juncas/udacity_p7_cat_vs_dog/blob/master/report.pdf)文件，该文件为毕业项目的详细报告。

猫狗大战的训练数据和测试数据可以在[这里](https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/data)下载。

## 文件说明            
- README.md: 本说明文件。

- [cat_and_dog_mergenet1.ipynb](https://github.com/juncas/udacity_p7_cat_vs_dog/blob/master/cat_and_dog_mergenet1.ipynb): 由Resnet50、Xception和Inception-V3三种模型构建融合模型的过程。

- [cat_and_dog_Inception_v3.ipynb](https://github.com/juncas/udacity_p7_cat_vs_dog/blob/master/cat_and_dog_Inception_v3.ipynb): Inception_v3模型的迁移学习过程。       

- [cat_and_dog_NASNet_transfer.ipynb](https://github.com/juncas/udacity_p7_cat_vs_dog/blob/master/cat_and_dog_NASNet_transfer.ipynb): NASNet模型的迁移学习过程。        

- [cat_and_dog_mergenet2.ipynb](https://github.com/juncas/udacity_p7_cat_vs_dog/blob/master/cat_and_dog_mergenet2.ipynb): 由Densenet169、Xception和Inception-V3三种模型构建融合模型的过程。

- [cat_and_dog_Densenet169_transfer.ipynb](https://github.com/juncas/udacity_p7_cat_vs_dog/blob/master/cat_and_dog_Densenet169_transfer.ipynb): Densenet169模型的迁移学习和Fine-tuning过程。     

- [cat_and_dog_Xception_transfer.ipynb](https://github.com/juncas/udacity_p7_cat_vs_dog/blob/master/cat_and_dog_Xception_transfer.ipynb): Xception模型的迁移学习和Fine-tuning过程。      

- [cat_and_dog_Resnet_transfer.ipynb](https://github.com/juncas/udacity_p7_cat_vs_dog/blob/master/cat_and_dog_Resnet_transfer.ipynb): Resnet50模型的迁移学习和Fine-tuning过程。        

- [cat_and_dog_mergenet3.ipynb](https://github.com/juncas/udacity_p7_cat_vs_dog/blob/master/cat_and_dog_mergenet3.ipynb): 由经过深度优化的Resnet50、Xception和Densenet169三种模型构建融合模型的过程。

- [data_set_statistics.ipynb](https://github.com/juncas/udacity_p7_cat_vs_dog/blob/master/data_set_statistics.ipynb): 在训练数据上做一些简单的统计统计      

- [helper.py](https://github.com/juncas/udacity_p7_cat_vs_dog/blob/master/resnext.py): 辅助函数基本上都在这个文件里。

- [resnext.py](https://github.com/juncas/udacity_p7_cat_vs_dog/blob/master/resnext.py): ResNeXt模型，来自[这里](https://github.com/titu1994/Keras-ResNeXt)。本项目没有使用到，但是出现了import列表里。

- [report.pdf](https://github.com/juncas/udacity_p7_cat_vs_dog/blob/master/report.pdf): 项目总结报告，涵盖了项目的说明和结果。


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
- GPU: 
	- device 0: GTX1070Ti-8G
	- device 1: Tesla K40c
- 其他配置不重要

## 操作系统
Ubuntu 16.04 64位

## 运行时间
### 特征提取时间(K40c)
- ResNet50: 1100s
- Xception: 1350s
- DenseNet169: 2200s
- Inception_V3: 1500s
- NASNetMobile: 1260s

### 训练时间
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
