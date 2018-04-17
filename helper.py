import os
import shutil
import numpy as np
import random
from tqdm import tqdm  
from time import time
from PIL import Image
import h5py
import pandas as pd

from keras.models import *
from keras.layers import *
from keras.applications import *
from keras.preprocessing.image import *
from keras.callbacks import *
from keras.optimizers import *
from keras.utils import *
from keras import backend as K

from sklearn.utils import shuffle


#为数据建立symbol link并划分为训练集和验证集
def prepare_data_file():    
    np.random.seed(2017)
    work_dir  = os.getcwd()
    train_dir = work_dir + "/train/"
    test_dir  = work_dir + "/test/"
    data_dir  = work_dir + "/data/"
    
    if(os.path.exists(data_dir)):
        shutil.rmtree(data_dir)
        
    split_train_dir = work_dir+"/data/train"
    split_valid_dir = work_dir+"/data/valid"
    split_test_dir  = work_dir+"/data/test"
    os.mkdir(data_dir)
    
    os.mkdir(split_train_dir)
    os.mkdir(split_train_dir+"/dog")
    os.mkdir(split_train_dir+"/cat")
    
    os.mkdir(split_valid_dir)
    os.mkdir(split_valid_dir+"/dog")
    os.mkdir(split_valid_dir+"/cat")
    
    os.mkdir(split_test_dir)
    os.mkdir(split_test_dir+"/test")
        
    train_files = os.listdir(train_dir)    
    num_train_files = len(train_files)
    file_index =  np.arange(num_train_files)
    
    np.random.shuffle(file_index)
    for i in tqdm(range(num_train_files)):
        file = train_files[file_index[i]]
        if i>=num_train_files*0.2:
            file_dir = split_train_dir
        else:
            file_dir = split_valid_dir
            
        if "dog" in file.split('.'):
            os.symlink(train_dir+file, file_dir+"/dog/"+file)
        else:
            os.symlink(train_dir+file, file_dir+"/cat/"+file)
    
    test_files = os.listdir(test_dir)    
    num_test_files = len(test_files)
    for i in tqdm(range(num_test_files)):
        file = test_files[i]
        os.symlink(test_dir+file, split_test_dir+"/test/"+file)
        
    return split_train_dir, split_valid_dir, split_test_dir

def write_feature_data(base_model, image_shape, train_dir, test_dir, batch_size, preprocess_input = None):
    input_tensor = Input((image_shape[0], image_shape[1], 3))
    x = input_tensor
    if preprocess_input:
        x = Lambda(preprocess_input)(x)
    
    base_model_instance = base_model(input_tensor=x, weights='imagenet', include_top=False)
    base_model_instance.save_weights(base_model_instance.name+'-imagenet.h5')
    
    model = Model(base_model_instance.input, GlobalAveragePooling2D()(base_model_instance.output))

    gen = ImageDataGenerator()
    train_generator = gen.flow_from_directory(train_dir, image_shape, shuffle=False, 
                                              batch_size=batch_size)
    test_generator = gen.flow_from_directory(test_dir, image_shape, shuffle=False, 
                                             batch_size=batch_size, class_mode=None)
    print(train_generator.samples)
    print(test_generator.samples)
    train_feature = model.predict_generator(train_generator, train_generator.samples, verbose=1)
    test_feature = model.predict_generator(test_generator, test_generator.samples, verbose=1)
    with h5py.File("feature_%s.h5"%base_model_instance.name) as h:
        h.create_dataset("train", data=train_feature)
        h.create_dataset("test", data=test_feature)
        h.create_dataset("label", data=train_generator.classes)
        

#从文件中读取特征向量和标签
def load_feature_data(filename):
    np.random.seed(2017)

    X_train = []
    X_test = []

    with h5py.File(filename, 'r') as h:
        X_train.append(np.array(h['train']))
        X_test.append(np.array(h['test']))
        Y_train = np.array(h['label'])

    X_train = np.concatenate(X_train, axis=1)
    X_test = np.concatenate(X_test, axis=1)

    X_train, Y_train = shuffle(X_train, Y_train)
    return X_train, Y_train, X_test

def predict_on_model(test_data_dir, X_test, model, model_name):
    y_test = model.predict(X_test, verbose=1)
    y_test = y_test.clip(min=0.005, max=0.995)
    df = pd.read_csv("sample_submission.csv")

    gen = ImageDataGenerator()
    test_generator = gen.flow_from_directory(test_data_dir, (224, 224), shuffle=False, 
                                             batch_size=16, class_mode=None)

    for i, fname in enumerate(test_generator.filenames):
        index = int(fname[fname.rfind('/')+1:fname.rfind('.')])
        df.set_value(index-1, 'label', y_test[i])

    df.to_csv(model_name, index=None)
    df.head(10)
    
def predict_on_resnet(n, width, heigth, test_data_dir, model, weight, output_name):
    x_test = np.zeros((n,width,heigth,3),dtype=np.uint8)

    for i in tqdm(range(n)):
        img = load_img(test_data_dir+"/test/"+'/%d.jpg' % (i+1))    
        x_test[i,:,:,:] = img_to_array(img.resize((width,heigth),Image.ANTIALIAS))
    
    model.load_weights(weight)
    y_test = model.predict(x_test, verbose=1)
    y_test = y_test.clip(min=0.005, max=0.995)
    
    df = pd.read_csv("sample_submission.csv")
    for i in tqdm(range(n)):
        df.set_value(i, 'label', y_test[i])
    df.to_csv(output_name, index=None)
    df.head(10)
    
def predict_on_xception(n, width, heigth, test_data_dir, model, weight, output_name):
    x_test = np.zeros((n,width,heigth,3),dtype=np.uint8)

    for i in tqdm(range(n)):
        img = load_img(test_data_dir+"/test/"+'/%d.jpg' % (i+1)) 
        x_test[i,:,:,:] = img_to_array(img.resize((width,heigth),Image.ANTIALIAS))
    
#     x_test = xception.preprocess_input(x_test)
    model.load_weights(weight)
    y_test = model.predict(x_test, verbose=1)
    y_test = y_test.clip(min=0.005, max=0.995)
    
    df = pd.read_csv("sample_submission.csv")
    for i in tqdm(range(n)):
        df.set_value(i, 'label', y_test[i])
    df.to_csv(output_name, index=None)
    df.head(10)
    
def predict_on_densenet(n, width, heigth, test_data_dir, model, weight, output_name):
    x_test = np.zeros((n,width,heigth,3),dtype=np.uint8)

    for i in tqdm(range(n)):
        img = load_img(test_data_dir+"/test/"+'/%d.jpg' % (i+1))    
        x_test[i,:,:,:] = img_to_array(img.resize((width,heigth),Image.ANTIALIAS))
    
    model.load_weights(weight)
    y_test = model.predict(x_test, verbose=1)
    y_test = y_test.clip(min=0.005, max=0.995)
    
    df = pd.read_csv("sample_submission.csv")
    for i in tqdm(range(n)):
        df.set_value(i, 'label', y_test[i])
    df.to_csv(output_name, index=None)
    df.head(10)

def write_fine_tuned_feature_data(base_model, image_shape, train_dir, test_dir, batch_size, weight, preprocess_input = None):
    x_input = Input((image_shape[0], image_shape[1], 3))
    if preprocess_input:
        x_input = Lambda(preprocess_input)(x_input)

    base_model = base_model(input_tensor=x_input, include_top=False, pooling = 'avg')

    for layer in base_model.layers:
        layer.trainable = False

    x = Dropout(0.5)(base_model.output)
    x = Dense(1, activation='sigmoid')(x)
    fine_tune_model = Model(base_model.input, x)
    
    fine_tune_model.load_weights(weight)
    print(fine_tune_model.layers[-3].name)
    feature_model = Model(fine_tune_model.input,fine_tune_model.layers[-3].output)

    gen = ImageDataGenerator()
    train_generator = gen.flow_from_directory(train_dir, image_shape, shuffle=False, 
                                              batch_size=batch_size)
    test_generator = gen.flow_from_directory(test_dir, image_shape, shuffle=False, 
                                             batch_size=batch_size, class_mode=None)
    print(train_generator.samples)
    print(test_generator.samples)
    train_feature = feature_model.predict_generator(train_generator, train_generator.samples, verbose=1)
    test_feature = feature_model.predict_generator(test_generator, test_generator.samples, verbose=1)
    with h5py.File("fine_tuned_feature_%s.h5"%base_model.name) as h:
        h.create_dataset("train", data=train_feature)
        h.create_dataset("test", data=test_feature)
        h.create_dataset("label", data=train_generator.classes)