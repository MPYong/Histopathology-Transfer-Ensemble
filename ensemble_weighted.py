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
pixel_sizes = [80, 120, 160]
labels = ['Normal', 'Abnormal']


#adjust parameter here
base_no_default = 3
mode = 'affine'
size = 160


model_ensemble_path = str(size) + ' result all\\' + str(size) + ' ensemble_weighted'

# Generate fold dataset
save_file = os.path.join('C:\\', str(size) + '_affine.npz')
data = np.load(save_file)
X_train = data['X_train']
X_test = data['X_test']
y_train = data['y_train']
y_test = data['y_test']



# generate new model


def get_model():
    base_model = InceptionV3(weights='imagenet')
    '''
    x = base_model.get_layer('conv4_block5_2_relu').output
    x = layers.Conv2D(32, 3, activation="relu", name = 'convo_histo')(x)
    x = layers.MaxPooling2D(3, name = 'localmax_histo')(x)
    prediction = layers.GlobalMaxPooling2D(name = 'globalmax_histo')(x)
    '''
    x = base_model.get_layer('avg_pool').output
    x = layers.Dense(256, name='fully_connected', activation='relu')(x)
    prediction = layers.Dense(n_class, name='output', activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=prediction)
    
    return model


    
 
def get_model_ensemble(n_model):
    x = keras.Input(shape=(n_model,))
    #y = layers.Dense(20, activation = 'relu')(x)
    #a = layers.Dense(5, activation = 'relu')(x)
    y = layers.Dense(1, activation = 'sigmoid')(x)
    model = Model(inputs=x, outputs=y)
    
    return model


def ensemble_predictions(members, testX, testy):
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
    return vote, result


def matrix(y_test, y_predict):
    matrix = confusion_matrix(y_test, y_predict)
    tn, fp, fn, tp = matrix.ravel()
    specificity = tn /(tn+fp)
    precision = tp / (tp+fp)
    recall = tp / (tp+fn)
    f1 = 2 * (precision * recall)/(precision+recall)
    return matrix, specificity, precision, recall, f1, tn, fp, fn, tp
    
# evaluate a specific number of members in an ensemble
def evaluate_n_members(members, testX, testy):
    # make prediction
    yhat, probability = ensemble_predictions(members, testX, testy)
    # calculate accuracy
    index_y = tensorflow.argmax(testy, axis=1)
    matrices, specificity, precision, recall, f1, tn, fp, fn, tp = matrix(index_y, yhat)
    return np.array([[accuracy_score(index_y, yhat), log_loss(index_y, probability), roc_auc_score(index_y, yhat), precision, recall, specificity, f1]]), yhat, probability


'''
#############################################################
#
#for existing ensemble model to get result again
#

a = np.array([])



for j, path_i in enumerate(model_base_path):
    n_class = num_model_ensemble = y_train.shape[1]
    num_model_base = len(path_i)
    
    # load base model
    model_base = list()
    
    for i in range(num_model_base):
        model_base.append(tensorflow.keras.models.load_model(os.path.join(path, path_i[i])))
    
    
    # get all base models output for testing
    result_base_test = np.empty((1, y_test.shape[0], y_test.shape[1]))
    for i in range(num_model_base):
        result_base_test = np.append(result_base_test, np.expand_dims(model_base[i].predict(X_test), axis = 0 ), axis = 0)
    result_base_test = result_base_test[1:]
    
    # load ensemble model
    model_ensemble = list()
    
    model_ensemble_path = str(size) + ' result all\\' + str(size) + ' ensemble_weighted ' + str(j+base_no)
    
    if mode == 'affine':
        model_ensemble_path += ' affine'
    
    
    for i in range(num_model_ensemble):
        model_ensemble.append(tensorflow.keras.models.load_model(os.path.join(path, model_ensemble_path, 'ensemble_' + str(i) + '.h5')))
    
    # get score
    ensemble_test_score = evaluate_n_members(model_ensemble, result_base_test, y_test)
    a = np.append(a, ensemble_test_score)
    
a = a.reshape(-1, 7)

'''

######################################################
#
# initiate and train ensemble model here
#

n_class = num_model_ensemble = y_val.shape[1]

# adjust ensemble training parameter
#-------------------------
num_model_base = 5
#-----------------------------

model_base = list()

for i in range(num_model_base):
    model_base.append(tensorflow.keras.models.load_model(os.path.join(path, model_base_path[i])))


# get all base models output for training ensemble
result_base_all = np.empty((1, y_val.shape[0], y_val.shape[1]))
for i in range(num_model_base):
    result_base_all = np.append(result_base_all, np.expand_dims(model_base[i].predict(X_val), axis = 0 ), axis = 0)
result_base_all = result_base_all[1:]
'''
# get all base models output for testing
result_base_test_all = np.empty((1, y_test.shape[0], y_test.shape[1]))
for i in range(num_model_base):
    result_base_test_all = np.append(result_base_test_all, np.expand_dims(model_base[i].predict(X_test), axis = 0 ), axis = 0)
result_base_test_all = result_base_test_all[1:]
    '''

for num_model_base in [3, 5]:
    
    # channge folder name here
    #----------------------------
    model_ensemble_path = str(size) + ' result all\\' + str(size) + ' ensemble_weighted'
    model_ensemble_path = model_ensemble_path + ' ' + str(num_model_base) + ' affine new'
    #------------------------------
    result_base = result_base_all[:num_model_base]


    '''
    if mode == 'affine':
        model_ensemble_path += ' affine'
    '''
    # load base model
    #model_base = list()
    
    #for i in range(num_model_base):
     #   model_base.append(tensorflow.keras.models.load_model(os.path.join(path, model_base_path[i])))
    
    '''
    for i in range(num_model_base):
        model_base.append(get_model())
        
    for i in range(num_model_base):
        model_base[i].compile(loss='categorical_crossentropy', optimizer='SGD', metrics=['accuracy'])
        model_base[i].fit(X_train, y_train, batch_size = 20, epochs = 50)
        model_base[i].save(os.path.join('GasHisSDB', '80_model_' + str(i) + '.h5'))
    '''
    '''
    # get all base models output for training ensemble
    result_base = np.empty((1, y_train.shape[0], y_train.shape[1]))
    for i in range(num_model_base):
        result_base = np.append(result_base, np.expand_dims(model_base[i].predict(X_train), axis = 0 ), axis = 0)
    result_base = result_base[1:]
    
    # get all base models output for testing
    result_base_test = np.empty((1, y_test.shape[0], y_test.shape[1]))
    for i in range(num_model_base):
        result_base_test = np.append(result_base_test, np.expand_dims(model_base[i].predict(X_test), axis = 0 ), axis = 0)
    result_base_test = result_base_test[1:]
    '''
    
    #create folder to store ensemble model
    os.makedirs(os.path.join(path, model_ensemble_path))
    
    
    # load ensemble model
    model_ensemble = list()
    
    '''
    for i in range(num_model_ensemble):
        model_ensemble.append(tensorflow.keras.models.load_model(os.path.join(path, model_ensemble_path, 'ensemble_' + str(i) + '.h5')))
    '''
    
    for i in range(num_model_ensemble):
        model_ensemble.append(get_model_ensemble(num_model_base))
        model_ensemble[i].save(os.path.join(path, model_ensemble_path, 'ensemble_' + str(i) + '.h5'))
    
    # train ensemble model
    for i in range(num_model_ensemble):
        X_train_ensemble = np.copy(result_base)
        X_train_ensemble = X_train_ensemble[:,:,i]
        X_train_ensemble = np.transpose(X_train_ensemble)
        
        y_train_ensemble = np.copy(y_val)
        y_train_ensemble = y_train_ensemble[:,i]
        
        model_ensemble[i].compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        model_ensemble[i].fit(X_train_ensemble, y_train_ensemble, epochs = 5,batch_size = 20)
        
        model_ensemble[i].save(os.path.join(path, model_ensemble_path, 'ensemble_' + str(i) + '.h5'))
    
    '''
    # get all base models output for testing
    result_base_test_all = np.empty((1, y_test.shape[0], y_test.shape[1]))
    for i in range(num_model_base):
        result_base_test_all = np.append(result_base_test_all, np.expand_dims(model_base[i].predict(X_test), axis = 0 ), axis = 0)
    result_base_test_all = result_base_test_all[1:]
    result_base_test = result_base_test_all[:num_model_base]
    
    
    
    # get score
    ensemble_test_score = evaluate_n_members(model_ensemble, result_base_test, y_test)
    '''

# test the weighted averaging ensemble model

zzz = np.array([])

assert model_base_path[0].startswith(str(X_test.shape[1]))
assert model_base_path_unarranged[0].startswith(str(X_test.shape[1]))
assert model_base_path_unarranged[0].find('affine') != -1
assert model_base_path[0].find('affine') != -1

# adjust ensemble training parameter
#-------------------------
num_model_base = 5
#-----------------------------

model_base = list()

for i in range(num_model_base):
    model_base.append(tensorflow.keras.models.load_model(os.path.join(path, model_base_path[i])))

'''
# get all base models output for training ensemble
result_base_all = np.empty((1, y_train.shape[0], y_train.shape[1]))
for i in range(num_model_base):
    result_base_all = np.append(result_base_all, np.expand_dims(model_base[i].predict(X_train), axis = 0 ), axis = 0)
result_base_all = result_base_all[1:]
'''
# get all base models output for testing
result_base_test_all = np.empty((1, y_test.shape[0], y_test.shape[1]))
for i in range(num_model_base):
    result_base_test_all = np.append(result_base_test_all, np.expand_dims(model_base[i].predict(X_test), axis = 0 ), axis = 0)
result_base_test_all = result_base_test_all[1:]

for num_model_base in range(3, 6, 2):
    n_class = num_model_ensemble = y_val.shape[1]
    
    #result_base = result_base_all[:num_model_base]
    result_base_test = result_base_test_all[:num_model_base]
    
    # load ensemble model
    model_ensemble = list()
    
    
    #change folder name here###########################
    #--------------------
    model_ensemble_path = str(size) + ' result all\\' + str(size) + ' ensemble_weighted ' + str(num_model_base) + ' affine new'
    #---------------------

    '''
    if mode == 'affine':
        model_ensemble_path += ' affine'
    '''
    
    for i in range(num_model_ensemble):
        model_ensemble.append(tensorflow.keras.models.load_model(os.path.join(path, model_ensemble_path, 'ensemble_' + str(i) + '.h5')))
    
    # get score
    ensemble_test_score, y_wa, probability_wa = evaluate_n_members(model_ensemble, result_base_test, y_test)
    zzz = np.append(zzz, ensemble_test_score)
    
zzz = zzz.reshape(-1, 7)
zzz *= 100

# add 'new' or not
#--------------------------------------------
np.savez(os.path.join(path,  str(size) + ' result all', str(size)  +  '_' + mode + '_weighted_table.npz'), table = zzz)
#--------------------------------------------

for size in [80,120,160]:
    table = np.load(os.path.join(path,  str(size) + ' result all', str(size)  +  '_' + mode + '_weighted_table.npz'))
    table = table['table']
    table *= 100
    np.savez(os.path.join(path,  str(size) + ' result all', str(size)  +  '_' + mode + '_weighted_table.npz'), table = table)

#ensemble_train_score = evaluate_n_members(model_ensemble, result_base, y_train)
'''
# make prediction
# testX [model, sample, class]
# testy [sample, class]
# result [sample, class]
result = np.empty((y_train.shape[0], 1))
for model in model_ensemble:
    X_train_ensemble = np.copy(result_base)
    X_train_ensemble = X_train_ensemble[:,:,i]
    X_train_ensemble = np.transpose(X_train_ensemble) #[sample, model]
        
    #y_train_ensemble = np.copy(testy)
    #y_train_ensemble = y_train_ensemble[:,i] #[sample]
        
    result = np.append(result, model.predict(X_train_ensemble), axis = 1)
    # [sample] -> [sample, class]
result = result[:,1:]

yhat = np.argmax(result, axis = 1)

# calculate accuracy
index_y = tensorflow.argmax(y_train, axis=1)
print('index',index_y)
print('yhay', yhat)
c = accuracy_score(index_y, yhat)
'''
'''
# get all base models output for testing
result_base_test = np.empty((1, y_test.shape[0], y_test.shape[1]))
for i in range(num_model_base):
    result_base_test = np.append(result_base_test, np.expand_dims(model_base[i].predict(X_test), axis = 0 ), axis = 0)
result_base_test = result_base_test[1:]

ensemble_test_score = evaluate_n_members(model_ensemble, result_base_test, y_test)

'''




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
