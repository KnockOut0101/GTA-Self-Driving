import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from tflearn.layers.normalization import local_response_normalization
import numpy as np


def alexnet(width, height, lr):
    network = input_data(shape=[None, width, height, 1], name='input')
    network = conv_2d(network, 96, 11, strides=4, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)
    network = conv_2d(network, 256, 5, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)
    network = conv_2d(network, 384, 3, activation='relu')
    network = conv_2d(network, 384, 3, activation='relu')
    network = conv_2d(network, 256, 3, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)
    network = fully_connected(network, 4096, activation='tanh')
    network = dropout(network, 0.5)
    network = fully_connected(network, 4096, activation='tanh')
    network = dropout(network, 0.5)
    network = fully_connected(network, 3, activation='softmax')
    network = regression(network, optimizer='momentum',
                         loss='categorical_crossentropy',
                         learning_rate=lr, name='targets')

    model = tflearn.DNN(network, checkpoint_path='model_alexnet',
                        max_checkpoints=1, tensorboard_verbose=2, tensorboard_dir='log')

    return model

WIDTH = 80
HEIGHT = 60
LR = 1e-3
EPOCH = 15
MODEL_NAME = 'pygta5_3-car-{}-{}-{}-epochs.model'.format(LR, 'alexnetv2', EPOCH)

model = alexnet(WIDTH,HEIGHT, LR)

train_data = np.load('AI_training_data_2.npy',None,1)

train = train_data[:-10000]
test = train_data[-10000:]

X = np.array([i[0] for i in train]).reshape(-1, WIDTH,HEIGHT, 1)
Y = [i[1] for i in train]

test_x = np.array([i[0] for i in test]).reshape(-1, WIDTH,HEIGHT, 1)
test_y = [i[0] for i in test]

model.fit({'input':X}, {'targets':Y}, n_epoch = EPOCH, validation_set=({'input':X},{'targets':Y}),
          snapshot_step = 500, show_metric = True , run_id = MODEL_NAME)

model.save(MODEL_NAME)
