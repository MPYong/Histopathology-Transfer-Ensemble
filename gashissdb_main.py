import glob, os
from PIL import Image
import numpy as np
from numpy import array, asarray, savetxt, save, load

import tensorflow
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input, decode_predictions
import numpy as np
from tensorflow.keras.applications import EfficientNetB0, EfficientNetB1, DenseNet121, DenseNet169, MobileNet, MobileNetV2, InceptionResNetV2, NASNetMobile, Xception

from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve,roc_auc_score, precision_recall_curve, confusion_matrix

import matplotlib.pyplot as plt

import pickle

import pandas as pd

from time import time

from tensorflow import keras
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve,roc_auc_score, confusion_matrix, log_loss

import cv2

import random

import gc

#from cbam import cbam_block, channel_attention, spatial_attention


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
physical_devices = tensorflow.config.list_physical_devices('GPU')
# print(physical_devices[0])
# print(tf.__version__)
tensorflow.config.experimental.set_memory_growth(physical_devices[0], True)

#import requests

#proxies = {"http":"http://1706352:930723-14-5539@192.168.30.238:8000"}



path = "E:\\yongmingping\\GasHisSDB"
path_dataset = 'G:\\'
#pixel_sizes = [80, 120, 160]
size = 160
labels = ['Normal', 'Abnormal']

epoch_total = 30
BATCH_SIZE = 20

model_names = ['mobilenet affine', 'mobilenetv2 affine', 'efficientnetb0 affine', 'efficientnetb1 affine',
               'densenet121 affine', 'densenet169 affine', 'inceptionv3 affine', 'xception affine']

mode = ''

'''
save_file = os.path.join(path, str(80) + '.npz')
data = np.load(save_file)
X_train = data['X_train'][:,:,:,:]
X_test = data['X_test'][:,:,:,:]
y_train = data['y_train'][:,:]
y_test = data['y_test'][:,:]
'''

def load_data(pixel_size, mode):
    #save_file = os.path.join(path_dataset, str(pixel_size) + '.npz')
    #save_file = os.path.join(path_dataset, str(pixel_size) + '_affine.npz')
    #save_file = os.path.join(path_dataset, str(pixel_size) + '_gan_complete.npz')

    if mode == 'normal':
        save_file = os.path.join(path_dataset, str(pixel_size) + '_new.npz')
    elif mode == 'affine':
        save_file = os.path.join(path_dataset, str(pixel_size) + '_affine_new.npz')
    assert save_file
    data = np.load(save_file)
    X_train = data['X_train']
    X_test = data['X_test']
    X_val = data['X_val']
    y_train = data['y_train']
    y_test = data['y_test']
    y_val = data['y_val']
    return X_train, X_test, X_val, y_train, y_test, y_val, save_file

def load_data_val(pixel_size, mode):
    #save_file = os.path.join(path_dataset, str(pixel_size) + '.npz')
    #save_file = os.path.join(path_dataset, str(pixel_size) + '_affine.npz')
    #save_file = os.path.join(path_dataset, str(pixel_size) + '_gan_complete.npz')
    save_file = os.path.join(path, str(pixel_size) + '_val_test.npz')
    assert save_file
    data = np.load(save_file)
    X_val = data['X_val']
    X_test = data['X_test']
    y_val = data['y_val']
    y_test = data['y_test']
    return X_val, X_test, y_val, y_test, save_file
    

def auc(model, y_test, y_predict, pixel_size):
    fpr , tpr , thresholds = roc_curve(y_test.argmax(axis=1) , y_predict.argmax(axis=1))
    
    plt.plot(fpr,tpr) 
    #plt.axis([0,1,0,1]) 
    plt.xlabel('False Positive Rate') 
    plt.ylabel('True Positive Rate') 
    #plt.savefig(os.path.join(path, str(pixel_size) + ' result all', str(pixel_size) + ' result ' + str(result_num),  'auc_' + str(pixel_size) + '_' + str(count + 1) +'.jpg'))
    #plt.show()    
    plt.figure().clf()
  
    print(len(y_test.argmax(axis=1)))
    auc_score=roc_auc_score(y_test.argmax(axis=1),y_predict.argmax(axis=1))
    
    
    
    return auc_score



def matrix(model, y_test, y_predict):
    matrix = confusion_matrix(y_test.argmax(axis=1), y_predict.argmax(axis=1))
    tn, fp, fn, tp = matrix.ravel()
    specificity = tn /(tn+fp)
    precision = tp / (tp+fp)
    recall = tp / (tp+fn)
    f1 = 2 * (precision * recall)/(precision+recall)
    
    
    return matrix, specificity, precision, recall, f1, tn, fp, fn, tp






def run(model_name, X_train, X_test, y_train, y_test, accuracy_epoch, loss_epoch, auc_epoch, pixel_size = 80, count_1 = 0):
    n_class = 2
    
    model_path = os.path.join(path, str(size) + ' result all', str(pixel_size) + ' ' + model_name, str(pixel_size) + '_' + str(count_1) + '.h5')
    
    if os.path.isfile(model_path):
         model = tensorflow.keras.models.load_model(model_path)
    else:



        '''
        #InceptionResNetV2
        base_model = InceptionResNetV2(weights='imagenet')
        x = base_model.get_layer('avg_pool').output
        prediction = layers.Dense(n_class, name='output', activation='softmax')(x)
        '''
        
        #NAS
        '''
        base_model = NASNetMobile(weights='imagenet', include_top =False)
        x = base_model.output
        y = layers.GlobalAveragePooling2D()(x)
        prediction = layers.Dense(n_class, name='output', activation='softmax')(y)
        '''
        if model_name.find('inceptionv3') != -1:
            base_model = InceptionV3(weights='imagenet')
        
            x = base_model.get_layer('avg_pool').output
            
            #x = base_model.get_layer('mixed0').output
            #x = cbam_block(x)
            #x = layers.GlobalAveragePooling2D()(x)
            #x = layers.Dropout(0.2)(x)
            #x = layers.Dense(36, name='fully_connected', activation='relu')(x)
            #x = layers.Dropout(0.2)(x)
            
            prediction = layers.Dense(n_class, name='output', activation='softmax')(x)
        
        elif model_name.find('xception') != -1:
            base_model = Xception(weights='imagenet', include_top =False, pooling = 'avg')
            x = base_model.output
            #y = layers.GlobalAveragePooling2D()(x)
            prediction = layers.Dense(n_class, name='output', activation='softmax')(x)
            
        
        # EfficientNetntation available 
        elif model_name.find('efficientnetb0') != -1:
            base_model = EfficientNetB0(weights='imagenet', include_top = False, input_shape=(160, 160, 3))
            #x = base_model.get_layer('top_dropout').output
            prediction = layers.Dense(n_class, name='output', activation='softmax')(x)
        
        elif model_name.find('efficientnetb1') != -1:
            base_model = EfficientNetB1(weights='imagenet')
            x = base_model.get_layer('top_dropout').output
            prediction = layers.Dense(n_class, name='output', activation='softmax')(x)
        
        elif model_name.find('densenet121') != -1:
            base_model = DenseNet121(weights='imagenet')
            x = base_model.get_layer('avg_pool').output
            prediction = layers.Dense(n_class, name='output', activation='softmax')(x)
            
        elif model_name.find('densenet169') != -1:
            base_model = DenseNet169(weights='imagenet')
            x = base_model.get_layer('avg_pool').output
            prediction = layers.Dense(n_class, name='output', activation='softmax')(x)
        
        elif model_name.find('mobilenetv2') != -1:
            base_model = MobileNetV2(weights='imagenet')
            x = base_model.get_layer('out_relu').output #reshape_2 - V1
            x = layers.GlobalAveragePooling2D()(x)
            prediction = layers.Dense(n_class, name='output', activation='softmax')(x)
        
        elif model_name.find('mobilenet') != -1:
            base_model = MobileNet(weights='imagenet')
            x = base_model.get_layer('dropout').output
            y = layers.Reshape((1024,))(x)
            prediction = layers.Dense(n_class, name='output', activation='softmax')(y)


        
        model = Model(inputs=base_model.input, outputs=prediction)
         
        
        model.save(model_path)
    
    model.compile(loss='categorical_crossentropy', optimizer='SGD', metrics=['accuracy'])
    
    # freeze the training of pretrained model
    '''
    for layer in base_model.layers:
        layer.trainable = False
    
    model.fit(X_train, y_train, batch_size = 20, epochs = 1)
    '''
    
    # unfreeze the training of pretrained model
    for layer in model.layers:
        layer.trainable = True
    
    '''
    if count < 3:
        for layer in model.layers:
            layer.trainable = False
            
        model.get_layer('output').trainable = True
    
    else:
        for layer in model.layers:
            layer.trainable = True
    '''
    
    table_train = np.array([])
    
    table_test = np.array([])
    
    for count in range (count_1, epoch_total):
    #for count in range (30):
        
        #model_path = os.path.join(path, str(size) + ' result all', str(pixel_size) + ' result ' + str(result_num), str(pixel_size) + '_' + str(count) + '.h5')
        #model = tensorflow.keras.models.load_model(model_path)
        
        model.fit(X_train, y_train, batch_size = BATCH_SIZE, epochs = 1)
        '''
        loss, accuracy = model.evaluate(X_train, y_train, batch_size = 20)
        accuracy_epoch = np.append(accuracy_epoch, accuracy)
        
        y_predict = model.predict(X_train)
        auc_score = auc(model, y_train, y_predict, pixel_size)
        auc_epoch = np.append(auc_epoch, auc_score)

        matrices, specificity, precision, recall, f1, tn, fp, fn, tp = matrix(model, y_train, y_predict)
        table_train = np.append(table_train, [[accuracy, loss, auc_score, precision, recall, specificity, f1]])
        
        del y_predict
        gc.collect()
        '''
        
        loss, accuracy = model.evaluate(X_test, y_test, batch_size = BATCH_SIZE)
        #accuracy_epoch = np.append(accuracy_epoch, accuracy)


        
        y_predict = model.predict(X_test)
        auc_score = auc(model, y_test, y_predict, pixel_size)
        auc_epoch = np.append(auc_epoch, auc_score)

        matrices, specificity, precision, recall, f1, tn, fp, fn, tp = matrix(model, y_test, y_predict)
        table_test = np.append(table_test, [[accuracy, loss, auc_score, precision, recall, specificity, f1]])
        
        
        
        model_path_new = os.path.join(path, str(size) + ' result all', str(pixel_size) + ' ' + model_name, str(pixel_size) + '_' + str(count+1) + '.h5')
        model.save(model_path_new)
        
        del y_predict
        gc.collect()
        
    del model
    gc.collect()
    tensorflow.keras.backend.clear_session()
    
    #table_train = table_train.reshape(-1, 7)
    table_test = table_test.reshape(-1, 7)
    
    return table_test

def evaluate_result_val(model_name, X_val, X_test, y_val, y_test, pixel_size = 80, count_1 = 0):
    table_val = np.array([])
    table_test = np.array([])
    
    
    for count in range (30):    
        model_path = os.path.join(path, str(size) + ' result all', str(pixel_size) + ' ' + model_name, str(pixel_size) + '_' + str(count + 1) + '.h5')
        model = tensorflow.keras.models.load_model(model_path)
        
        
        loss, accuracy = model.evaluate(X_val, y_val, batch_size = 20)

        y_predict = model.predict(X_val)
        auc_score = auc(model, y_val, y_predict, pixel_size)

        matrices, specificity, precision, recall, f1, tn, fp, fn, tp = matrix(model, y_val, y_predict)
        table_val = np.append(table_val, [[accuracy, loss, auc_score, precision, recall, specificity, f1]])
        
        del y_predict
        gc.collect()
        
        table_val = table_val.reshape(-1, 7)
        
        loss, accuracy = model.evaluate(X_test, y_test, batch_size = 20)

        y_predict = model.predict(X_test)
        auc_score = auc(model, y_test, y_predict, pixel_size)

        matrices, specificity, precision, recall, f1, tn, fp, fn, tp = matrix(model, y_test, y_predict)
        table_test = np.append(table_test, [[accuracy, loss, auc_score, precision, recall, specificity, f1]])
        
        del y_predict
        gc.collect()
        
        table_test = table_test.reshape(-1, 7)
    
    return table_val, table_test

def evaluate_result_test(X_test, y_test, model_name, epoch, pixel_size = 80):
    
    model_path = os.path.join(path, str(size) + ' result all', str(pixel_size) + ' ' + model_name, str(pixel_size) + '_' + str(epoch) + '.h5')
    
    model = tensorflow.keras.models.load_model(model_path)
    
    loss, accuracy = model.evaluate(X_test, y_test, batch_size = 20)

    y_predict = model.predict(X_test)
    auc_score = auc(model, y_test, y_predict, pixel_size)

    matrices, specificity, precision, recall, f1, tn, fp, fn, tp = matrix(model, y_test, y_predict)
    table_test = [accuracy, loss, auc_score, precision, recall, specificity, f1]
    
    del y_predict
    gc.collect()
    
    #table_test = table_test.reshape(-1, 7)
    
    return table_test



#-----------------------load dataset, change 'affine' or 'normal'
X_train, X_test, X_val, y_train, y_test, y_val, save_file = load_data(size, 'affine')
#-----------------------

#X_val, X_test, y_val, y_test, save_file = load_data_val(size, 'affine')
'''
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=1, stratify=y_test)
save_file = os.path.join(path, str(size) + '_val_test' + '.npz')
np.savez(save_file, X_val = X_val, X_test = X_test, y_val = y_val, y_test = y_test)
'''
'''
a1 = y_test.sum(axis=0)
a2 = y_train.sum(axis=0)
a_total = a1+a2
'''
'''
X_train /= 255
X_test /= 255
y_test /= 255
y_train /= 255

save_file = os.path.join(path, str(size) + '_affine_normalized_new.npz')
np.savez(save_file, X_train = X_train, X_test =  X_test, y_train = y_train, y_test = y_test)
'''
model_names = ['mobilenet affine', 'mobilenetv2 affine', 'efficientnetb0 affine', 'efficientnetb1 affine',
               'densenet121 affine', 'densenet169 affine', 'inceptionv3 affine', 'xception affine']

for model_name in model_names:
    # create folder and train model
    result_num = len(os.listdir(os.path.join(path, str(size) + ' result all'))) + 1
    #aaa
    #result_num -=1
    os.makedirs(os.path.join(path
                             , str(size) + ' result all', str(size) + ' ' + model_name))
    accuracy_epoch = np.array([])
    loss_epoch = np.array([])
    auc_epoch = np.array([])
    
    
    #start training
    table_val = run(model_name, X_train, X_val, y_train, y_val, 
                             accuracy_epoch, loss_epoch, auc_epoch, pixel_size = size)
    
    np.savez(os.path.join(path,  str(size) + ' result all', str(size) + ' ' + model_name, str(size) + '_table_metric' + '.npz'), table_val = table_val)


'''
#continue training
accuracy_epoch, loss_epoch, auc_epoch = run(X_train, X_test, y_train, y_test, 
                         accuracy_epoch, loss_epoch, auc_epoch, pixel_size = size, count_1 = 18)
'''



#no need for now
#evaluate model on validation and testing set
for model_name in model_names:
    accuracy_epoch = np.array([])
    loss_epoch = np.array([])
    auc_epoch = np.array([])

    table_val, table_test = evaluate_result_val(model_name, X_val, X_test, y_val, y_test,  pixel_size = size, count_1 = 0)
    
    np.savez(os.path.join(path,  str(size) + ' result all', str(size) + ' ' + model_name, str(size) + '_val_test' + '.npz'), table_val = table_val, table_test = table_test)



'''
# check image
img = np.copy(image_input[2545])
img = img.astype(np.uint8)
img = Image.fromarray(img, 'RGB')
img.save(os.path.join(path, 'saveit.jpg'))
'''


################################################
#
# generate accuracy list for models
#
'''
model_names = ['mobilenet', 'mobilenetv2',
'efficientnetb0', 'efficientnetb1',
               'densenet121', 'densenet169', 'inceptionv3', 'xception']
mode = 'normal'
'''

'''
model_names = ['mobilenet dcgan', 'mobilenetv2 dcgan', 'efficientnetb0 dcgan', 'efficientnetb1 dcgan', 
               'densenet121 dcgan', 'densenet169 dcgan', 'inceptionv3 dcgan', 'xception dcgan']
'''

model_names = ['mobilenet affine' , 'mobilenetv2 affine',
               'efficientnetb0 affine', 'efficientnetb1 affine',
    'densenet121 affine', 'densenet169 affine', 'inceptionv3 affine', 'xception affine']

mode = 'affine'

assert (save_file.find('affine') != -1) == (model_names[0].find('affine') != -1)

def evaluate_result_with_name(model_name, X_train, X_test, y_train, y_test, accuracy_epoch, loss_epoch, auc_epoch, accuracy_epoch_train, loss_epoch_train, auc_epoch_train, pixel_size = 80, count_1 = 0):
    for count in range (30):
        model_path = os.path.join(path, str(size) + ' result all', str(pixel_size) + ' ' + model_name, str(pixel_size) + '_' + str(count + 1) + '.h5')
        model = tensorflow.keras.models.load_model(model_path)

        loss, accuracy = model.evaluate(X_test, y_test, batch_size = 20)
        accuracy_epoch = np.append(accuracy_epoch, accuracy)
        loss_epoch = np.append(loss_epoch, loss)
            
        y_predict = model.predict(X_test)
        auc_score = auc(model, y_test, y_predict, pixel_size, count)
        auc_epoch = np.append(auc_epoch, auc_score)
        
        #-----------------------------@@@@@@@@@@@@@@@@@@
        loss_train, accuracy_train = model.evaluate(X_train, y_train, batch_size = 20)
        accuracy_epoch_train = np.append(accuracy_epoch_train, accuracy_train)
        loss_epoch_train = np.append(loss_epoch_train, loss_train)
            
        y_predict = model.predict(X_train)
        auc_score_train = auc(model, y_train, y_predict, pixel_size, count)
        auc_epoch_train = np.append(auc_epoch_train, auc_score_train)
        
        del model, y_predict
        gc.collect()
        tensorflow.keras.backend.clear_session()
    
    return accuracy_epoch, loss_epoch, auc_epoch, accuracy_epoch_train, loss_epoch_train, auc_epoch_train

# generate new test and train data result
for model_name in model_names:
    accuracy_epoch = np.array([])
    loss_epoch = np.array([])
    auc_epoch = np.array([])
    
    accuracy_epoch_train = np.array([])
    loss_epoch_train = np.array([])
    auc_epoch_train = np.array([])

    accuracy_epoch, loss_epoch, auc_epoch, accuracy_epoch_train, loss_epoch_train, auc_epoch_train = evaluate_result_with_name(model_name, X_train, X_test, y_train, y_test, 
                             accuracy_epoch, loss_epoch, auc_epoch, accuracy_epoch_train, loss_epoch_train, auc_epoch_train, pixel_size = size, count_1 = 0)
    
    np.savez(os.path.join(path,  str(size) + ' result all', str(size) + ' ' + model_name,str(size) + '_accuracy_new' + '.npz'), accuracy_epoch = accuracy_epoch)
    np.savez(os.path.join(path,  str(size) + ' result all', str(size) + ' ' + model_name,str(size) + '_loss_new' + '.npz'), loss_epoch = loss_epoch)
    np.savez(os.path.join(path,  str(size) + ' result all', str(size) + ' ' + model_name,str(size) + '_auc_new' + '.npz'), auc_epoch = auc_epoch)
    
    np.savez(os.path.join(path,  str(size) + ' result all', str(size) + ' ' + model_name,str(size) + '_accuracy_training_new' + '.npz'), accuracy_epoch = accuracy_epoch_train)
    np.savez(os.path.join(path,  str(size) + ' result all', str(size) + ' ' + model_name,str(size) + '_loss_training_new' + '.npz'), loss_epoch = loss_epoch_train)
    np.savez(os.path.join(path,  str(size) + ' result all', str(size) + ' ' + model_name,str(size) + '_auc_training_new' + '.npz'), auc_epoch = auc_epoch_train)
    
    del accuracy_epoch, loss_epoch, auc_epoch, accuracy_epoch_train, loss_epoch_train, auc_epoch_train
    gc.collect()
    
    
'''
def evaluate_result_with_name_train(model_name, X_train, X_test, y_train, y_test, accuracy_epoch, loss_epoch, auc_epoch, pixel_size = 80, count_1 = 0):
    for count in range (30):
        model_path = os.path.join(path, str(size) + ' result all', str(pixel_size) + ' ' + model_name, str(pixel_size) + '_' + str(count + 1) + '.h5')
        model = tensorflow.keras.models.load_model(model_path)

        loss, accuracy = model.evaluate(X_train, y_train, batch_size = 20)
        accuracy_epoch = np.append(accuracy_epoch, accuracy)
        loss_epoch = np.append(loss_epoch, loss)
            
        y_predict = model.predict(X_train)
        auc_score = auc(model, y_train, y_predict, pixel_size, count)
        auc_epoch = np.append(auc_epoch, auc_score)
        
        del model, y_predict
        tensorflow.keras.backend.clear_session()
    
    return accuracy_epoch, loss_epoch, auc_epoch


# generate new training data result for training vs testing accuracy graph
for model_name in model_names:
    accuracy_epoch = np.array([])
    loss_epoch = np.array([])
    auc_epoch = np.array([])

    accuracy_epoch, loss_epoch, auc_epoch = evaluate_result_with_name_train(model_name, X_train, X_test, y_train, y_test, 
                             accuracy_epoch, loss_epoch, auc_epoch, pixel_size = size, count_1 = 0)
    np.savez(os.path.join(path,  str(size) + ' result all', str(size) + ' ' + model_name,str(size) + '_accuracy_training_new' + '.npz'), accuracy_epoch = accuracy_epoch)
    np.savez(os.path.join(path,  str(size) + ' result all', str(size) + ' ' + model_name,str(size) + '_loss_training_new' + '.npz'), loss_epoch = loss_epoch)
    np.savez(os.path.join(path,  str(size) + ' result all', str(size) + ' ' + model_name,str(size) + '_auc_training_new' + '.npz'), auc_epoch = auc_epoch)
'''


################
#
#
#-----------------change model between normal or afine
model_names = ['mobilenet affine' , 'mobilenetv2 affine',
               'efficientnetb0 affine', 'efficientnetb1 affine',
    'densenet121 affine', 'densenet169 affine', 'inceptionv3 affine', 'xception affine']

mode = 'affine'
#-----------------


assert (save_file.find('affine') != -1) == (model_names[0].find('affine') != -1)

'''
#run this to get result table
ccc = np.array([])

for model_name in model_names:
    aaa = np.load(os.path.join(path, str(size) + ' result all\\' + str(size) +' ' + model_name + '\\' + str(size) + '_accuracy_new.npz'))
    bbb = aaa['accuracy_epoch']
    ddd = np.load(os.path.join(path, str(size) + ' result all\\' + str(size) +' ' + model_name + '\\' + str(size) + '_loss_new.npz'))
    fff = ddd['loss_epoch']
    eee = np.load(os.path.join(path, str(size) + ' result all\\' + str(size) +' ' + model_name + '\\' + str(size) + '_auc_new.npz'))
    ggg = eee['auc_epoch']
    ccc =np.append(ccc, [bbb.argmax(axis=0)+1, bbb[bbb.argmax(axis=0)], fff[bbb.argmax(axis=0)], ggg[bbb.argmax(axis=0)]])
ccc = ccc.reshape(-1, 4)


a = np.array([])
for model_name, epoch_no in zip(model_names, ccc[:,0]):
    model_path = os.path.join(path, str(size) + ' result all\\' +
                          str(size)+ ' '+ model_name + '\\' + str(size) + '_' +
                          str(int(epoch_no)) + '.h5')
    model = tensorflow.keras.models.load_model(model_path)
    y_predict = model.predict(X_test)
    matrices, specificity, precision, recall, f1, tn, fp, fn, tp = matrix(model, y_test, y_predict)
    a = np.append(a, [[precision, recall, specificity, f1]])
    #del model
a = a.reshape(-1,4)
b = np.append(ccc, a, axis = 1)

np.savez(os.path.join(path,  str(size) + ' result all', str(size)  + '_' + mode + '_table.npz'), table = b)
'''

size = 160
mode = 'affine'
X_train, X_test, X_val, y_train, y_test, y_val, save_file = load_data(size, 'affine')

b = np.array([])
c = np.array([])
d = np.array([])
table_test = np.array([])

for model_name in model_names:
    
    aaa = np.load(os.path.join(path, str(size) + ' result all\\' + str(size) +' ' +  model_name + '\\' + str(size) + '_table_metric.npz'))
    '''
    bbb = aaa['table_test']
    bbb = bbb * 100
    
    b =np.append(b, [bbb[:,0].argmax(axis=0)+1, bbb[bbb[:,0].argmax(axis=0),0], bbb[bbb[:,0].argmax(axis=0),1], bbb[bbb[:,0].argmax(axis=0),2],
                         bbb[bbb[:,0].argmax(axis=0),3], bbb[bbb[:,0].argmax(axis=0),4],bbb[bbb[:,0].argmax(axis=0),5],bbb[bbb[:,0].argmax(axis=0),6]])
    '''
    ccc = aaa['table_val']
    ccc = ccc * 100
    '''
    c =np.append(c, [ccc[:,0].argmax(axis=0)+1, ccc[ccc[:,0].argmax(axis=0),0], ccc[ccc[:,0].argmax(axis=0),1], ccc[ccc[:,0].argmax(axis=0),2],
                         ccc[ccc[:,0].argmax(axis=0),3], ccc[ccc[:,0].argmax(axis=0),4],ccc[ccc[:,0].argmax(axis=0),5],ccc[ccc[:,0].argmax(axis=0),6]])
    best_epoch = ccc[:,0].argmax(axis=0)+1
    '''
    c =np.append(c, [ccc[:,2].argmax(axis=0)+1, ccc[ccc[:,2].argmax(axis=0),0], ccc[ccc[:,2].argmax(axis=0),1], ccc[ccc[:,2].argmax(axis=0),2],
                         ccc[ccc[:,2].argmax(axis=0),3], ccc[ccc[:,2].argmax(axis=0),4],ccc[ccc[:,2].argmax(axis=0),5],ccc[ccc[:,2].argmax(axis=0),6]])
    best_epoch = ccc[:,2].argmax(axis=0)+1
    
    table_test = np.append(table_test, evaluate_result_test(X_test, y_test, model_name, epoch = best_epoch, pixel_size = size))
    
    '''
    #testing accuracy corresponding to best validation accuracy
    d = np.append(d, [ccc[:,0].argmax(axis=0)+1, bbb[ccc[:,0].argmax(axis=0),0], bbb[ccc[:,0].argmax(axis=0),1], bbb[ccc[:,0].argmax(axis=0),2],
                        bbb[ccc[:,0].argmax(axis=0),3],bbb[ccc[:,0].argmax(axis=0),4],bbb[ccc[:,0].argmax(axis=0),5],bbb[ccc[:,0].argmax(axis=0),6]])
'''
    
    
#b = b.reshape(-1,8)
c = c.reshape(-1,8)
#d = d.reshape(-1,8)
table_test = table_test.reshape(-1,7)
table_test *= 100

np.savez(os.path.join(path,  str(size) + ' result all', str(size)  + '_' + mode + '_table_new.npz'), table_val = c, table_test = table_test)

size = 80
table_load = np.load(os.path.join(path,  str(size) + ' result all', str(size)  + '_' + mode + '_table_new.npz'))
dataval = table_load['table_val']
datatest = table_load['table_test']

#--------------- change model between original and affine version
model_names = ['mobilenet affine' , 'mobilenetv2 affine',
               'efficientnetb0 affine', 'efficientnetb1 affine',
    'densenet121 affine', 'densenet169 affine', 'inceptionv3 affine', 'xception affine']

mode = 'affine'
#---------------
model_labels = ['MobileNet', 'MobileNetV2', 'EfficientnetB0', 'EfficientnetB1', 
               'DenseNet121', 'DenseNet169', 'InceptionV3', 'Xception']

assert (save_file.find('affine') != -1) == (model_names[0].find('affine') != -1)

#generate training vs testing accuracy graph
for model_name, model_label in zip (model_names, model_labels):
    
    aaa = np.load(os.path.join(path, str(size) + ' result all\\' + str(size) +' ' + model_name + '\\' + str(size) + '_table_metric.npz'))
    bbb = aaa['table_test']
    bbb = bbb[:,0]
    epoch_len = range(len(bbb))
    epoch_len = [s + 1 for s in epoch_len]
    bbb = bbb * 100
    plt.plot(epoch_len, bbb, label = 'Testing')
    
    aaa_training = np.load(os.path.join(path, str(size) + ' result all\\' + str(size) +' ' + model_name + '\\' + str(size) + '_table_metric.npz'))
    bbb_training = aaa_training['table_train']
    bbb_training = bbb_training[:,0]
    bbb_training = bbb_training * 100
    plt.plot(epoch_len, bbb_training, label =  'Training')

    
    plt.xlabel('Number of Epochs')
    plt.ylabel('Accuracy (%)')
    #-----------change title between original or augmented dataset
    plt.title('Training vs testing accuracy of ' + model_label + '\nfor ' + str(size) + '-pixels sub-database augmented dataset')
    #-----------------
    
    plt.legend()
    #plt.figure(figsize=(1000,1000))
    #-----------change title between original or augmented dataset
    plt.savefig(os.path.join(path, str(size) +  '_' + model_name + '_' + mode + '_accuracy.jpg'), dpi = 300)
    #-----------------
    #plt.show()
    plt.figure().clf()
    
size = 120
    
for i, [model_name, model_label] in enumerate(zip (model_names, model_labels)):
    
    plt.subplot(3,3, i+1)
    
    aaa = np.load(os.path.join(path, str(size) + ' result all\\' + str(size) +' ' + model_name + '\\' + str(size) + '_table_metric.npz'))
    bbb = aaa['table_test']
    bbb = bbb[:,0]
    epoch_len = range(len(bbb))
    epoch_len = [s + 1 for s in epoch_len]
    bbb = bbb * 100
    plt.plot(epoch_len, bbb, label = 'Testing')
    
    aaa_training = np.load(os.path.join(path, str(size) + ' result all\\' + str(size) +' ' + model_name + '\\' + str(size) + '_table_metric.npz'))
    bbb_training = aaa_training['table_train']
    bbb_training = bbb_training[:,0]
    bbb_training = bbb_training * 100
    plt.plot(epoch_len, bbb_training, label =  'Training')

    
    plt.xlabel('Number of Epochs')
    plt.ylabel('Accuracy (%)')
    
    plt.yticks(np.arange(85, 101, step=5))
    #-----------change title between original or augmented dataset
    plt.title(model_label)
    #-----------------
    
    plt.legend()
plt.tight_layout()
#plt.figure(figsize=(1000,1000))
#-----------change title between original or augmented dataset
plt.savefig(os.path.join(path, str(size) +  '_' + mode + '_accuracy.jpg'), dpi = 1000)
#-----------------
#plt.show()
plt.figure().clf()






#################################################
# get table and sorted model path 
#


size = 160

X_train, X_test, X_val, y_train, y_test, y_val, save_file = load_data(size, 'affine')

#-----------------change model between normal or affine
model_names = ['mobilenet affine', 'mobilenetv2 affine', 'efficientnetb0 affine', 'efficientnetb1 affine',
    'densenet121 affine', 'densenet169 affine', 'inceptionv3 affine', 'xception affine']
mode= 'affine'
#-----------------

table_original = np.load(os.path.join(path,  str(size) + ' result all', str(size)  + '_' + mode + '_table_new.npz'))
table_original = table_original['table_val']


#table_original = np.load(os.path.join(path,  str(size) + ' result all', str(size)  + '_' + mode + '_table.npz'))
#table_original = table_original['table']

#model_names_epoch = model_names.copy()
model_count = np.arange(len(model_names))
model_count = model_count[:, np.newaxis]
model_count = np.append(model_count, table_original, axis = 1)
model_count_2 = model_count[np.argsort(model_count[:, 2])]
model_count_2 = model_count_2[::-1]

complete_list = []
for i, item in enumerate(model_count):
    complete_list.append(model_names[int(item[0])] + '\\' + str(size) + '_' + str(int(item[1])) + '.h5')


top_list = []
for i, item in enumerate(model_count_2):
    top_list.append(model_names[int(item[0])] + '\\' + str(size) + '_' + str(int(item[1])) + '.h5')


model_base_path = top_list[:]
model_base_path_unarranged = complete_list[:]

model_base_path = [str(size) +' result all\\' + str(size) + ' ' + s for s in model_base_path]
model_base_path_unarranged = [str(size) +' result all\\' + str(size) + ' ' + s for s in model_base_path_unarranged]






###########################################
#
# generate intersect and union samples for gradcam purpose
#

def comparison_indices(y_test, y_predict):
    y_predict_2 = np.zeros_like(y_predict)
    index_list_correct = np.array([])
    index_list_wrong = np.array([])
    row_no = y_test.shape[0]
    y_predict_2[np.arange(len(y_predict)), y_predict.argmax(1)] = 1
    for count in range(row_no):
        if (np.array_equal(y_test[count], y_predict_2[count])):
            index_list_correct = np.append(index_list_correct, count)
        if not(np.array_equal(y_test[count], y_predict_2[count])):
            index_list_wrong = np.append(index_list_wrong, count)
        
    
    del y_predict_2
    gc.collect()
    
    return index_list_correct, index_list_wrong


assert model_base_path[0].startswith(str(X_test.shape[1]))

#common wrongly predicted samples across top 5 models (1)
all_correct_index = {'i':np.array([]), 'u':np.array([])}
all_wrong_index = {'i':np.array([]), 'u':np.array([])}


for k, specific_path in enumerate(model_base_path):
    model_path = os.path.join(path, specific_path)
    model = tensorflow.keras.models.load_model(model_path)
    y_predict = model.predict(X_test)
    index_list_correct, index_list_wrong = comparison_indices(y_test, y_predict)
    
    
    #intersect (2)
    if (k>0):
        all_correct_index['i'] = np.intersect1d(all_correct_index['i'], index_list_correct)
        all_wrong_index['i'] = np.intersect1d(all_wrong_index['i'], index_list_wrong)
    else:
        all_correct_index['i'] = np.copy(index_list_correct)
        all_wrong_index['i'] = np.copy(index_list_wrong)
        
    #union (2)
    if (k>0):
        all_correct_index['u'] = np.append(all_correct_index['u'], index_list_correct)
        all_wrong_index['u'] = np.append(all_wrong_index['u'], index_list_wrong)
    else:
        all_correct_index['u'] = np.copy(index_list_correct)
        all_wrong_index['u'] = np.copy(index_list_wrong)
    all_correct_index['u'] = np.unique(all_correct_index['u']) 
    all_wrong_index['u'] = np.unique(all_wrong_index['u']) 
    
'''
np.random.shuffle(all_correct_index['i'])
np.random.shuffle(all_correct_index['i'])
np.random.shuffle(all_wrong_index['i'])
np.random.shuffle(all_wrong_index['u'])

'''
all_correct_index['i'] = all_correct_index['i'].astype('int')
all_correct_index['u'] = all_correct_index['u'].astype('int')
all_wrong_index['i'] = all_wrong_index['i'].astype('int')
all_wrong_index['u'] = all_wrong_index['u'].astype('int')

path_save_predict = (os.path.join(path, 'prediction', str(size)))

# save wrong
for symbol in ['i', 'u']:
    for k in all_wrong_index[symbol]:
        img = np.copy(X_test[k])
        img = img.astype(np.uint8)
        img = Image.fromarray(img, 'RGB')
        label = y_test[k].argmax()
            
        if symbol == 'i':  
            img.save(os.path.join(path_save_predict, 'wrong', 'intersect', str(k) + '_' + str(label) +'.png'))
        elif symbol == 'u':
            img.save(os.path.join(path_save_predict, 'wrong', 'union', str(k) + '_' + str(label) +'.png'))

# save correct
for symbol in ['i', 'u']:
    for k in all_correct_index[symbol]:
        img = np.copy(X_test[k])
        img = img.astype(np.uint8)
        img = Image.fromarray(img, 'RGB')
        label = y_test[k].argmax()
            
        if symbol == 'i':  
            img.save(os.path.join(path_save_predict, 'correct', 'intersect', str(k) + '_' + str(label) +'.png'))
        elif symbol == 'u':
            img.save(os.path.join(path_save_predict, 'correct', 'union', str(k) + '_' + str(label) +'.png'))    












####################################################
# Generate ROC curve
#
########################################
# Generate ROC 
#



# weighted averaging ensemble models
def ensemble_predictions_weighted(members, testX, testy, num_model_ensemble):
    # testX [model, sample, class]  
    # testy [sample, class]
    # result [sample, class]
    result = np.empty((testy.shape[0], 1))
    for i in range(num_model_ensemble):
        X_train_ensemble = np.copy(testX)
        X_train_ensemble = X_train_ensemble[:,:,i]
        X_train_ensemble = np.transpose(X_train_ensemble) #[sample, model]
        #y_train_ensemble = np.copy(testy)
        #y_train_ensemble = y_train_ensemble[:,i] #[sample]
        
        result = np.append(result, members[i].predict(X_train_ensemble), axis = 1) 
        # [sample] -> [sample, class]
    result = result[:,1:]

    vote = np.argmax(result, axis = 1)
    
    y_predict = np.zeros_like(result)
    y_predict[np.arange(len(result)), result.argmax(1)] = 1
    
    return y_predict, vote

# evaluate a specific number of members in an ensemble
def evaluate_n_members_weighted(members, testX, testy, num_model_ensemble):
    # make prediction
    y_predict, vote = ensemble_predictions_weighted(members, testX, testy, num_model_ensemble)

    return y_predict


#Average voting (Ensemble)
def ensemble_predictions(members, testX, testy):
    result = np.empty((1, testy.shape[0], testy.shape[1]))
    for model in members:
        result = np.append(result, np.expand_dims(model.predict(testX), axis = 0 ), axis = 0)
    result = result[1:]
    
    result_average = np.average(result, axis = 0)
    vote = np.argmax(result_average, axis = 1)
    
    
    y_predict = np.zeros_like(result_average)
    y_predict[np.arange(len(result_average)), result_average.argmax(1)] = 1
    
    return y_predict, vote


#Majority voting (Ensemble)
def ensemble_predictions_1(members, testX, testy):
    result = np.empty((1, testy.shape[0], testy.shape[1]))
    for model in members:
        result = np.append(result, np.expand_dims(model.predict(testX), axis = 0 ), axis = 0)
    result = result[1:]
    
    m,n = result.shape[:2]
    I,J = np.ogrid[:m,:n]
    
    result_max = np.zeros_like(result)
    result_max[I, J, result.argmax(2)] = 1
    summed = np.sum(result_max, axis=0)
    vote = np.argmax(summed, axis = 1)

    y_predict = np.zeros_like(summed)
    y_predict[np.arange(len(summed)), summed.argmax(1)] = 1
    
    return y_predict,vote


# evaluate a specific number of members in an ensemble (Ensemble)
def evaluate_n_members(members, testX, testy, mode):
    # make prediction
    if mode == 0:
        y_predict, vote = ensemble_predictions(members, testX, testy)
    elif (mode == 1):
        y_predict, vote = ensemble_predictions_1(members, testX, testy)
        
    return y_predict
    


def auc_all(model_base_path_unarranged, model_labels, model_base_path, y_test, X_test, pixel_size, title_add = ''):
    
    
    ijk = 0
    y_predict_total = np.empty((1, y_test.shape[0], y_test.shape[1]))
    #pretrained networks
    for specific_path, model_label in zip(model_base_path_unarranged, model_labels):
        model_path = os.path.join(path, specific_path)
        model = tensorflow.keras.models.load_model(model_path)
        print(model_path)
        y_predict = model.predict(X_test)
        
        fpr , tpr , thresholds = roc_curve(y_test.argmax(axis=1) , y_predict.argmax(axis=1))
        
        y_predict_total = np.append(y_predict_total, np.expand_dims(y_predict, axis = 0 ), axis = 0)
        
        plt.plot(fpr,tpr, label = model_label)
        #plt.axis([0,1,0,1])
    
    # weighted averaging ensemble models
    for count in range(3,6,2):
        
        #num_model_base = len(path_i)
        num_model_base = count
    
        # load base model
        model_base = list()
        
        for i in range(num_model_base):
            #model_base.append(tensorflow.keras.models.load_model(os.path.join(path, path_i[i])))
            model_base.append(tensorflow.keras.models.load_model(os.path.join(path, model_base_path[i])))
        
        
        result_base_test = np.empty((1, y_test.shape[0], y_test.shape[1]))
        for i in range(num_model_base):
            result_base_test = np.append(result_base_test, np.expand_dims(model_base[i].predict(X_test), axis = 0 ), axis = 0)
        result_base_test = result_base_test[1:]
        
        # load ensemble model
        num_model_ensemble = y_test.shape[1]
        
        model_ensemble = list()
        
        model_ensemble_path = str(pixel_size) + ' result all\\' + str(pixel_size) + ' ensemble_weighted ' + str(num_model_base) + ' ' + title_add
        
        for i in range(num_model_ensemble):
            model_ensemble.append(tensorflow.keras.models.load_model(os.path.join(path, model_ensemble_path, 'ensemble_' + str(i) + '.h5')))
        
    
        y_predict = evaluate_n_members_weighted(model_ensemble, result_base_test, y_test, num_model_ensemble)
            
        y_predict_total = np.append(y_predict_total, np.expand_dims(y_predict, axis = 0 ), axis = 0)
        
        fpr , tpr , thresholds = roc_curve(y_test.argmax(axis=1) , y_predict.argmax(axis=1))
        
        plt.plot(fpr,tpr, label = 'Weighted averaging ensemble model (' + str(num_model_base) + ')')
            
    
    # unweighted/majority voting ensemble models
    for mode_index in range(2):
        for count in range(3,6,2):
    
            #num_model_base = len(path_i)
            num_model_base = count
        
            # load base model
            model_base = list()
            
            for i in range(num_model_base):
                #model_base.append(tensorflow.keras.models.load_model(os.path.join(path, path_i[i])))
                model_base.append(tensorflow.keras.models.load_model(os.path.join(path, model_base_path[i])))
            
            #ensemble_train_score = evaluate_n_members(model_base, X_train, y_train)
            y_predict = evaluate_n_members(model_base, X_test, y_test, mode = mode_index)
            
            y_predict_total = np.append(y_predict_total, np.expand_dims(y_predict, axis = 0 ), axis = 0)
            
            
            fpr , tpr , thresholds = roc_curve(y_test.argmax(axis=1) , y_predict.argmax(axis=1))
            
            if mode_index == 0:
                label = 'Unweighted averaging ensemble model (' + str(num_model_base) + ')'
            elif mode_index == 1:
                label = 'Majority voting ensemble model (' + str(num_model_base) + ')'
            
            plt.plot(fpr,tpr, label = label)
        
    
    #clean up
    '''
    plt.plot(np.arange(0,2), 'r--')
    
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.savefig(os.path.join(path, str(pixel_size) +  '_' + mode + '_roc.jpg'), dpi = 300)
    #plt.show()    
    plt.figure().clf()
    '''
    
    
    return y_predict_total

#
#  
###############################################

model_labels = ['MobileNet', 'MobileNetV2', 'EfficientnetB0', 'EfficientnetB1', 
               'DenseNet121', 'DenseNet169', 'InceptionV3', 'Xception']

all_labels = ['MobileNet', 'MobileNetV2', 'EfficientnetB0', 'EfficientnetB1', 
               'DenseNet121', 'DenseNet169', 'InceptionV3', 'Xception', 
               'Weighted averaging ensemble model (3)', 'Weighted averaging ensemble model (5)',
               'Unweighted averaging ensemble model (3)', 'Unweighted averaging ensemble model (5)',
               'Majority voting ensemble model (3)', 'Majority voting ensemble model (5)']





#assert model_base_path_unarranged[0].find('affine') != -1
#assert model_base_path[0].find('affine') != -1
assert (save_file.find('affine') != -1) == (model_base_path[0].find('affine') != -1)
assert (save_file.find('affine') != -1) == (model_base_path_unarranged[0].find('affine') != -1)
assert model_base_path[0].startswith(str(X_test.shape[1]))
assert model_base_path_unarranged[0].startswith(str(X_test.shape[1]))
#assert save_file.find('affine') != -1

#------------------ change title_add
y_predict_total = auc_all(model_base_path_unarranged, model_labels, model_base_path, y_test, X_test, size, title_add = 'affine new')
#------------------

y_predict_total = y_predict_total[1:]

np.savez(os.path.join(path,  str(size) + ' result all', str(size) + '_' + mode + '_all_predict' + '.npz'), y_predict_total = y_predict_total)



#-------------------------change size parameter
size = 160
mode = 'affine'
#-------------------------

###### load y_predict
y_predict_total = np.load(os.path.join(path,  str(size) + ' result all', str(size) + '_' + mode + '_all_predict' + '.npz'))
y_predict_total = y_predict_total['y_predict_total']
####---------

#-----------------------load dataset, change 'affine' or 'normal'
X_train, X_test, y_train, y_test, save_file = load_data(size, 'affine')
#-----------------------


auc_complete = np.array([])

table = np.load(os.path.join(path,  str(size) + ' result all', str(size)  + '_' + mode + '_table.npz'))
table = table['table']
auc_complete = np.append(auc_complete, table[:,3])

weighted_table = np.load(os.path.join(path,  str(size) + ' result all', str(size)  +  '_' + mode + '_weighted_table.npz'))
weighted_table = weighted_table['table']
auc_complete = np.append(auc_complete, weighted_table[:,2])

voting_table = np.load(os.path.join(path,  str(size) + ' result all', str(size)  +  '_' + mode + '_voting_table.npz'))
voting_table = voting_table['table']
auc_complete = np.append(auc_complete, voting_table[:,2])


'''
# load prediction table instead of starting over again
#--------------- change between original and affine dataset
predict_table = np.load(os.path.join(path,  str(size) + ' result all', str(size) + '_all_predict' + '.npz'))
#------------------
y_predict_total = predict_table['y_predict_total']
'''

plt.figure(figsize=(10, 10))
for y_predict, model_label, auc_individual in zip(y_predict_total, all_labels, auc_complete):

    fpr , tpr , thresholds = roc_curve(y_test.argmax(axis=1), y_predict.argmax(axis=1))
    
    plt.plot(fpr,tpr, label = model_label + '(AUC = ' + str(round(auc_individual,3)) + ')')
    
plt.plot(np.arange(0,2), 'r--')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
#----------------
plt.title('ROC for ' + str(size) + '-pixels sub-database')
#----------------
plt.legend()
#plt.figure(figsize=(1000,1000))
plt.savefig(os.path.join(path, str(size) +  '_' + mode + '_roc.jpg'), dpi = 600)
#plt.show()
plt.figure().clf()



plt.figure(figsize=(10, 10))
for y_predict, model_label in zip(y_predict_total, all_labels):

    precision , recall , thresholds = precision_recall_curve(y_test.argmax(axis=1), y_predict.argmax(axis=1))
    
    plt.plot(recall,precision, label = model_label)

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve for ' + str(size) + '-pixels sub-database')
plt.legend()
#plt.figure(figsize=(1000,1000))
plt.savefig(os.path.join(path, str(size) +  '_' + mode + '_precision_recall.jpg'), dpi = 600)
#plt.show()
plt.figure().clf()


import pandas as pd
import seaborn as sn
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

#plt.figure(figsize = (10,7))

all_labels = ['MobileNet', 'MobileNetV2', 'EfficientnetB0', 'EfficientnetB1', 
               'DenseNet121', 'DenseNet169', 'InceptionV3', 'Xception', 
               'Weighted averaging\nensemble model (3)', 'Weighted averaging\nensemble model (5)',
               'Unweighted averaging\nensemble model (3)', 'Unweighted averaging\nensemble model (5)',
               'Majority voting\nensemble model (3)', 'Majority voting\nensemble model (5)']


font = {'weight' : 'normal',
    'size'   : 4}
plt.rc('font', **font)


f, axes = plt.subplots(5, 3, figsize = (4,6))
plt.subplots_adjust(top = 0.9, bottom= 0.2, hspace=0.95, wspace = 0.7)
for i, [y_predict, model_label] in enumerate(zip(y_predict_total, all_labels)):
    
    #fig = plt.figure()
    
    matrices = confusion_matrix(y_test.argmax(axis=1), y_predict.argmax(axis=1))
    #plt.matshow(matrices)
    '''
    #plt.title('Confusion Matrix of ' + model_label + ' for\n' + str(size) + '-pixels sub-database')
    plt.title(model_label)
    #plt.colorbar()
    plt.ylabel('True Label')
    plt.xlabel('Predicated Label')
    '''
    classes = ['Normal', 'Abnormal']
    '''
    df_cfm = pd.DataFrame(matrices, index = classes, columns = classes)
    
    cfm_plot = sn.heatmap(df_cfm, annot=True, fmt='g')
        
    if (i <= 15):
        plt.subplot(3, 5, i+1)
    '''
    #else:
        #plt.subplot(2,5, i - 4+1)
        
    disp = ConfusionMatrixDisplay(matrices,
                                  display_labels=classes)
    disp.plot(ax=axes[int(i/3), i%3], xticks_rotation=0)
    disp.ax_.set_title(model_label, fontsize=5, fontweight = 'bold')
    disp.im_.colorbar.remove()
    disp.ax_.set_xlabel('Predicted class')
    #if i!=0:
    disp.ax_.set_ylabel('True class')

        
    
    #plt.savefig(os.path.join(path, str(size) + '_confusion_matrix_'+ model_label +'.jpg'), dpi = 600)
    
    #plt.figure().clf()
f.delaxes(axes[4,2])    
f.tight_layout()
plt.savefig(os.path.join(path, str(size) + '_confusion_matrix_'+ '.jpg'), dpi = 1000)
plt.figure().clf()


'''
#check if image is ok
seeagain = X_train[0]
seeagain = seeagain.astype(np.uint8)
img = Image.fromarray(seeagain, 'RGB')
img.save(os.path.join('E:\\yongmingping\\GasHisSDB\\wowthis.png'))

'''
'''
## convert your array into a dataframe
df = pd.DataFrame (ccc)

## save to xlsx file

filepath = os.path.join(path, str(size) + '.xlsx')

df.to_excel(filepath, index=False)



abc = tensorflow.keras.models.load_model('E:\yongmingping\\GasHisSDB\\80 result all\\80 efficientnetb0\\80_1.h5')
'''
