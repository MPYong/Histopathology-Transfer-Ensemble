# -*- coding: utf-8 -*-
"""
Created on Wed Dec 15 20:40:33 2021

@author: student
"""

import glob, os
from PIL import Image
import numpy as np
from numpy import array, asarray, savetxt, save, load

import tensorflow
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input, decode_predictions
import numpy as np

from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve,roc_auc_score, confusion_matrix, log_loss

import pickle

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
physical_devices = tensorflow.config.list_physical_devices('GPU')
# print(physical_devices[0])
# print(tf.__version__)
tensorflow.config.experimental.set_memory_growth(physical_devices[0], True)



path = "E:\\yongmingping\\GasHisSDB"
path_c = 'C:\\'
pixel_sizes = [80, 120, 160]
labels = ['Normal', 'Abnormal']


size = 160



# Generate fold dataset
save_file = os.path.join(path_c, str(size) + '_new.npz')
data = np.load(save_file)
X_train = data['X_train']
X_test = data['X_test']
y_train = data['y_train']
y_test = data['y_test']



#Average voting
def ensemble_predictions(members, testX, testy):
    result = np.empty((1, testy.shape[0], testy.shape[1]))
    for model in members:
        result = np.append(result, np.expand_dims(model.predict(testX), axis = 0 ), axis = 0)
    result = result[1:]
    
    result_average = np.average(result, axis = 0)
    vote = np.argmax(result_average, axis = 1)
    
    
    return vote, result_average


#Majority voting
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
    return vote, summed/num_model_base


def matrix(y_test, y_predict):
    matrix = confusion_matrix(y_test, y_predict)
    tn, fp, fn, tp = matrix.ravel()
    specificity = tn /(tn+fp)
    precision = tp / (tp+fp)
    recall = tp / (tp+fn)
    f1 = 2 * (precision * recall)/(precision+recall)
    return matrix, specificity, precision, recall, f1, tn, fp, fn, tp
    
# evaluate a specific number of members in an ensemble
def evaluate_n_members(members, testX, testy, mode):
    # make prediction
    if mode == 0:
        yhat, probability = ensemble_predictions(members, testX, testy)
    elif (mode == 1):
        yhat, probability = ensemble_predictions_1(members, testX, testy)
    # calculate accuracy
    index_y = tensorflow.argmax(testy, axis=1)
    matrices, specificity, precision, recall, f1, tn, fp, fn, tp = matrix(index_y, yhat)
    return np.array([[accuracy_score(index_y, yhat), log_loss(index_y, probability), roc_auc_score(index_y, yhat), precision, recall, specificity, f1]])#, yhat, probability




#ensemble_model.compile(loss='categorical_crossentropy', optimizer='SGD', metrics=['accuracy'])
#ensemble_model.evaluate(X_test, y_test, batch_size = 20)

'''
# freeze the training of pretrained model
for layer in base_model.layers:
    layer.trainable = False

model.fit(X_train, y_train, batch_size = 20, epochs = 5)


# unfreeze the training of pretrained model
for layer in base_model.layers:
    layer.trainable = True
'''



#ensemble_model.fit(X_train, y_train, batch_size = 20, epochs = 3)

#ensemble_model.evaluate(X_test, y_test, batch_size = 20)


'''
b = np.array([])

#average voting
#for path_i in model_base_path:
for count in range(3,6,2):

    #num_model_base = len(path_i)
    num_model_base = count
    
    
    
    if ((num_model_base % 2) != 0):
        # load base model
        model_base = list()
        
        for i in range(num_model_base):
            #model_base.append(tensorflow.keras.models.load_model(os.path.join(path, path_i[i])))
            model_base.append(tensorflow.keras.models.load_model(os.path.join(path, model_base_path[i])))
        
        #ensemble_train_score = evaluate_n_members(model_base, X_train, y_train)
        ensemble_test_score = evaluate_n_members(model_base, X_val, y_val, mode = 0)
        
        b = np.append(b, ensemble_test_score)
    
    
#majority voting
#for path_i in model_base_path:
for count in range(3,6,2):

    #num_model_base = len(path_i)
    num_model_base = count
    
    if ((num_model_base % 2) != 0):
    
        # load base model
        model_base = list()
        
        for i in range(num_model_base):
            #model_base.append(tensorflow.keras.models.load_model(os.path.join(path, path_i[i])))
            model_base.append(tensorflow.keras.models.load_model(os.path.join(path, model_base_path[i])))
        
        #ensemble_train_score = evaluate_n_members(model_base, X_train, y_train)

        ensemble_test_score = evaluate_n_members(model_base, X_val, y_val, mode = 1)
            
        b = np.append(b, ensemble_test_score)

    
b = b.reshape(-1, 7)
b *= 100

'''


a = np.array([])

#average voting
#for path_i in model_base_path:
for count in range(3,6,2):

    #num_model_base = len(path_i)
    num_model_base = count
    
    
    
    if ((num_model_base % 2) != 0):
        # load base model
        model_base = list()
        
        for i in range(num_model_base):
            #model_base.append(tensorflow.keras.models.load_model(os.path.join(path, path_i[i])))
            model_base.append(tensorflow.keras.models.load_model(os.path.join(path, model_base_path[i])))
        
        #ensemble_train_score = evaluate_n_members(model_base, X_train, y_train)
        ensemble_test_score = evaluate_n_members(model_base, X_test, y_test, mode = 0)
        
        a = np.append(a, ensemble_test_score)
    
    
#majority voting
#for path_i in model_base_path:
for count in range(3,6,2):

    #num_model_base = len(path_i)
    num_model_base = count
    
    if ((num_model_base % 2) != 0):
    
        # load base model
        model_base = list()
        
        for i in range(num_model_base):
            #model_base.append(tensorflow.keras.models.load_model(os.path.join(path, path_i[i])))
            model_base.append(tensorflow.keras.models.load_model(os.path.join(path, model_base_path[i])))
        
        #ensemble_train_score = evaluate_n_members(model_base, X_train, y_train)

        ensemble_test_score = evaluate_n_members(model_base, X_test, y_test, mode = 1)
            
        a = np.append(a, ensemble_test_score)

    
a = a.reshape(-1, 7)
a *= 100

np.savez(os.path.join(path,  str(size) + ' result all', str(size)  +  '_' + mode + '_voting_table.npz'), table = a)


for size in [80,120,160]:
    table = np.load(os.path.join(path,  str(size) + ' result all', str(size)  +  '_' + mode + '_voting_table.npz'))
    table = table['table']
    table *= 100
    np.savez(os.path.join(path,  str(size) + ' result all', str(size)  +  '_' + mode + '_voting_table.npz'), table = table)


'''
X_test_1 = X_test[0:7]
y_test_1 = y_test[0:7]
result = np.empty((1, y_test_1.shape[0], y_test_1.shape[1]))
for i in model:
    result = np.append(result, np.expand_dims(i.predict(X_test_1), axis = 0 ), axis = 0)
result = result[1:]
    
m,n = result.shape[:2]
I,J = np.ogrid[:m,:n]

result_max = np.zeros_like(result)
result_max[I, J, result.argmax(2)] = 1
summed = np.sum(result_max, axis=0)
vote = np.argmax(summed, axis = 1)
'''
'''
exclude = summed.argmax(0)
mask = np.ones(summed.shape, bool)
mask[exclude] = False
summed[mask] = 0
mask = np.invert(mask)
summed[mask] = 1
'''

#ensemble_model.save('80.h5')
