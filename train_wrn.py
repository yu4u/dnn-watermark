import numpy as np
import pandas as pd
import sys
import json
import os
import sklearn.metrics as metrics
import wide_residual_network as wrn
import keras.callbacks as callbacks
import keras.utils.np_utils as kutils
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from keras.optimizers import SGD
from keras.callbacks import LearningRateScheduler

from watermark_regularizers import WatermarkRegularizer
from watermark_regularizers import get_wmark_regularizers
from watermark_regularizers import show_encoded_wmark

RESULT_PATH = './result'
MODEL_CHKPOINT_FNAME = os.path.join(RESULT_PATH, 'WRN-Weights.h5')

def update_hdf5(fname, path, data):
    store = pd.HDFStore(fname)

    if path in store.keys():
        store.remove(path)
    store.append(path, data)
    store.close()

def save_wmark_signatures(prefix, model):
    for layer_id, wmark_regularizer in get_wmark_regularizers(model):
        fname_w = prefix + '_layer{}_w.npy'.format(layer_id)
        fname_b = prefix + '_layer{}_b.npy'.format(layer_id)
        np.save(fname_w, wmark_regularizer.get_matrix())
        np.save(fname_b, wmark_regularizer.get_signature())

lr_schedule = [60, 120, 160]  # epoch_step

def schedule(epoch_idx):
    if (epoch_idx + 1) < lr_schedule[0]:
        return 0.1
    elif (epoch_idx + 1) < lr_schedule[1]:
        return 0.02 # lr_decay_ratio = 0.2
    elif (epoch_idx + 1) < lr_schedule[2]:
        return 0.004
    return 0.0008

if __name__ == '__main__':
    settings_json_fname = sys.argv[1]
    train_settings = json.load(open(settings_json_fname))
    
    if not os.path.isdir(RESULT_PATH):
        os.makedirs(RESULT_PATH)
        
    # load dataset and fitting data for learning
    if train_settings['dataset'] == 'cifar10':
        dataset = cifar10
        nb_classes = 10
    else:
        print('not supported dataset "{}"'.format(train_settings['dataset']))
        exit(1)

    (trainX, trainY), (testX, testY) = dataset.load_data()
    trainX = trainX.astype('float32')
    trainX /= 255.0
    testX = testX.astype('float32')
    testX /= 255.0
    trainY = kutils.to_categorical(trainY)
    testY = kutils.to_categorical(testY)

    generator = ImageDataGenerator(rotation_range=10,
                                   width_shift_range=5./32,
                                   height_shift_range=5./32,
                                   horizontal_flip=True)
    generator.fit(trainX, seed=0, augment=True)

    if 'replace_train_y' in train_settings and len(train_settings['replace_train_y']) > 0:
        print('trainY was replaced from "{}"'.format(train_settings['replace_train_y']))
        trainY = np.load(train_settings['replace_train_y'])

    # read parameters
    batch_size = train_settings['batch_size']
    nb_epoch = train_settings['epoch']
    scale = train_settings['scale']
    embed_dim = train_settings['embed_dim']
    N = train_settings['N']
    k = train_settings['k']

    target_blk_id = train_settings['target_blk_id']
    base_modelw_fname = train_settings['base_modelw_fname']
    wtype = train_settings['wmark_wtype']
    randseed = train_settings['randseed'] if 'randseed' in train_settings else 'none'
    ohist_fname = train_settings['history']
    hist_hdf_path = 'WTYPE_{}/DIM{}/SCALE{}/N{}K{}B{}EPOCH{}/TBLK{}'.format(
        wtype, embed_dim, scale, N, k, batch_size, nb_epoch, target_blk_id).replace('.', '_')
    modelname_prefix = os.path.join(RESULT_PATH, 'wrn_' + hist_hdf_path.replace('/', '_'))

    # initialize process for Watermark
    b = np.ones((1, embed_dim))
    wmark_regularizer = WatermarkRegularizer(scale, b, wtype=wtype, randseed=randseed)

    init_shape = (3, 32, 32) if K.image_dim_ordering() == 'th' else (32, 32, 3)
    model = wrn.create_wide_residual_network(init_shape, nb_classes=nb_classes, N=N, k=k, dropout=0.00,
                                             wmark_regularizer=wmark_regularizer, target_blk_num=target_blk_id)
    model.summary()
    print('Watermark matrix:\n{}'.format(wmark_regularizer.get_matrix()))

    # training process
    sgd = SGD(lr=0.1, momentum=0.9, nesterov=True)
    model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["acc"])
    if len(base_modelw_fname) > 0:
        model.load_weights(base_modelw_fname)
    print("Finished compiling")

    hist = \
    model.fit_generator(generator.flow(trainX, trainY, batch_size=batch_size), samples_per_epoch=len(trainX), nb_epoch=nb_epoch,
                        callbacks=[callbacks.ModelCheckpoint(MODEL_CHKPOINT_FNAME, monitor="val_acc", save_best_only=True),
                                   LearningRateScheduler(schedule=schedule)
                        ],
                        validation_data=(testX, testY),
                        nb_val_samples=testX.shape[0],)
    show_encoded_wmark(model)

    # validate training accuracy
    yPreds = model.predict(testX)
    yPred = np.argmax(yPreds, axis=1)
    yPred = kutils.to_categorical(yPred)
    yTrue = testY

    accuracy = metrics.accuracy_score(yTrue, yPred) * 100
    error = 100 - accuracy
    print("Accuracy : ", accuracy)
    print("Error : ", error)

    # write history and model parameters to file
    update_hdf5(ohist_fname, hist_hdf_path, pd.DataFrame(hist.history))
    model.save_weights(modelname_prefix + '.weight')

    # write watermark matrix and embedded signature to file
    if target_blk_id > 0:
        save_wmark_signatures(modelname_prefix, model)