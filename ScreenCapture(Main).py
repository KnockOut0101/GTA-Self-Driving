from PIL import ImageGrab
import cv2
import numpy as np
import time
from numpy import ones,vstack
from numpy.linalg import lstsq
from statistics import mean
##import sys
##sys.path.insert(0,'G:\Kshitij\Studies\Dissertations\OpenCV')
import ctypes
import pyautogui
import win32gui, win32ui, win32con, win32api
import os
import pandas as pd
from collections import Counter
from random import shuffle
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from tflearn.layers.normalization import local_response_normalization

##input from keys or mouse

SendInput = ctypes.windll.user32.SendInput

W = 0x11
A = 0x1E
S = 0x1F
D = 0x20

# C struct redefinitions 
PUL = ctypes.POINTER(ctypes.c_ulong)
class KeyBdInput(ctypes.Structure):
    _fields_ = [("wVk", ctypes.c_ushort),
                ("wScan", ctypes.c_ushort),
                ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", PUL)]

class HardwareInput(ctypes.Structure):
    _fields_ = [("uMsg", ctypes.c_ulong),
                ("wParamL", ctypes.c_short),
                ("wParamH", ctypes.c_ushort)]

class MouseInput(ctypes.Structure):
    _fields_ = [("dx", ctypes.c_long),
                ("dy", ctypes.c_long),
                ("mouseData", ctypes.c_ulong),
                ("dwFlags", ctypes.c_ulong),
                ("time",ctypes.c_ulong),
                ("dwExtraInfo", PUL)]

class Input_I(ctypes.Union):
    _fields_ = [("ki", KeyBdInput),
                 ("mi", MouseInput),
                 ("hi", HardwareInput)]

class Input(ctypes.Structure):
    _fields_ = [("type", ctypes.c_ulong),
                ("ii", Input_I)]

# Actuals Functions

def PressKey(hexKeyCode):
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    ii_.ki = KeyBdInput( 0, hexKeyCode, 0x0008, 0, ctypes.pointer(extra) )
    x = Input( ctypes.c_ulong(1), ii_ )
    ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))

def ReleaseKey(hexKeyCode):
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    ii_.ki = KeyBdInput( 0, hexKeyCode, 0x0008 | 0x0002, 0, ctypes.pointer(extra) )
    x = Input( ctypes.c_ulong(1), ii_ )
    ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))

W = 0x11
A = 0x1E
S = 0x1F
D = 0x20

## KeyMapping 

keyList = ["\b"]
for char in "ABCDEFGHIJKLMNOPQRSTUVWXYZ 123456789,.'APS$\\":
    keyList.append(char)

def key_check():
    keys = []
    for key in keyList:
        if win32api.GetAsyncKeyState(ord(key)):
            keys.append(key)
    return keys

def keys_to_output(keys):
    output = [0,0,0]

    if 'A' in keys:
        output[0] = 1
    elif 'D' in keys:
        output[2] = 1
    else:
        output[1] = 1

    return output

'''Neural Network In different file'''

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
EPOCH = 13
MODEL_NAME = 'pygta5_2-car-{}-{}-{}-epochs.model'.format(LR, 'alexnetv2', EPOCH)
##
##model = alexnet(WIDTH,HEIGHT, LR)
##
##train_data = np.load('training_data-vid2.npy')
##
##train = train_data[:-100]
##test = train_data[-100:]
##
##X = np.array([i[0] for i in train]).reshape(-1, WIDTH,HEIGHT, 1)
##Y = [i[1] for i in train]
##
##test_x = np.array([i[0] for i in test]).reshape(-1, WIDTH,HEIGHT, 1)
##test_y = [i[0] for i in test]
##
##model.fit({'input':X}, {'targets':Y}, n_epoch = EPOCH, validation_set=({'input':X},{'targets':Y}),
##          snapshot_step = 500, show_metric = True , run_id = MODEL_NAME)
##
##model.save(MODEL_NAME)



def grab_screen(region = None):

    hwin = win32gui.GetDesktopWindow()

    if region:
        left, top, x2, y2 = region
        width = x2 - left + 1
        height = y2 - top + 1
    else:
        width = win32api.GetSystemMetrics(win32con.SM_CXVIRTUALSCREEN)
        height = win32api.GetSystemMetrics(win32con.SM_CYVIRTUALSCREEN)
        left = win32api.GetSystemMetrics(win32con.SM_XVIRTUALSCREEN)
        top = win32api.GetSystemMetrics(win32con.SM_YVIRTUALSCREEN)

    hwindc = win32gui.GetWindowDC(hwin)
    srcdc = win32ui.CreateDCFromHandle(hwindc)
    memdc = srcdc.CreateCompatibleDC()
    bmp = win32ui.CreateBitmap()
    bmp.CreateCompatibleBitmap(srcdc, width, height)
    memdc.SelectObject(bmp)
    memdc.BitBlt((0, 0), (width,height), srcdc, (left,top), win32con.SRCCOPY)

    signedIntsArray = bmp.GetBitmapBits(True)
    img = np.fromstring(signedIntsArray, dtype='uint8')
    img.shape = (height, width, 4)

    srcdc.DeleteDC()
    memdc.DeleteDC()
    win32gui.ReleaseDC(hwin, hwindc)
    win32gui.DeleteObject(bmp.GetHandle())

    return cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)



##def roi(img, vertices):
##    mask = np.zeros_like(img)
##    cv2.fillPoly(mask, vertices, 255)
##    masked = cv2.bitwise_and(img, mask)
##    return masked
##
##def draw_lanes(img,lines, color = [0, 255, 255], thickness = 3):
##    try:
##        ys=[]
##        for i in lines:
##            for ii in i:
##                ys += [ii[1],ii[3]]
##        min_y = min(ys)
##        max_y = 600
##        new_lines = []
##        line_dict = {}
##
##        for idx, i in enumerate(lines):
##            for xyxy in i:
##                x_coords = (xyxy[0],xyxy[2])
##                y_coords = (xyxy[1],xyxy[3])
##                A = vstack([x_coords, ones(len(x_coords))]).T
##                m, b = lstsq(A , y_coords)[0]
##
##                x1 = (min_y - b)/ m
##                x2 = (max_y - b)/ m
##
##                line_dict[idx] = [m,b,[int(x1) , min_y, int(x2), max_y]]
##                new_lines.append([int(x1), min_y, int(x2), max_y])
##
##        final_lanes = {}
##
##        for idx in line_dict:
##            final_lanes_copy = final_lanes.copy()
##            m = line_dict[idx][0]
##            b = line_dict[idx][1]
##            line = line_dict[idx][2]
##
##            if len(final_lanes) == 0:
##                final_lanes[m] = [[m,b,line]]
##
##            else:
##                found_copy = False
##
##                for other_ms in final_lanes_copy:
##
##                    if not found_copy:
##                        if abs(other_ms*1.2) > abs(m) > abs(other_ms*0.8):
##                            if abs(final_lanes_copy[other_ms][0][1]*1.2) > abs(b) > abs(final_lanes_copy[other_ms][0][1]*0.8):
##                                final_lanes[other_ms].append([m,b,line])
##                                found_copy = True
##                                break
##                            else:
##                                final_lanes[m] = [[m,b,line]]
##
##        line_counter = {}
##
##        for lanes in final_lanes:
##            line_counter[lanes] = len(final_lanes[lanes])
##
##        top_lanes = sorted(line_counter.items(), key=lambda item: item[1])[::-1][:2]
##
##        lane1_id = top_lanes[0][0]
##        lane2_id = top_lanes[1][0]
##
##        def average_lane(lane_data):
##            x1s = []
##            y1s = []
##            x2s = []
##            y2s = []
##            for data in lane_data:
##                x1s.append(data[2][0])
##                y1s.append(data[2][1])
##                x2s.append(data[2][2])
##                y2s.append(data[2][3])
##            return int(mean(x1s)), int(mean(y1s)), int(mean(x2s)), int(mean(y2s))
##
##        l1_x1, l1_y1, l1_x2, l1_y2 = average_lane(final_lanes[lane1_id])
##        l2_x1, l2_y1, l2_x2, l2_y2 = average_lane(final_lanes[lane2_id])
##
##        return [l1_x1, l1_y2, l1_x2, l1_y2], [l2_x1, l2_y1, l2_x2, l2_y2], lane1_id, lane2_id
##    except Exception as e:
##        print(str(e))
##
##def process_img(image):
##    original_image = image
##    processed_img = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
##    processed_img = cv2.Canny(processed_img, threshold1=200, threshold2=300)
##    processed_img = cv2.GaussianBlur(processed_img, (7,7), 0)
##    vertices = np.array([[10,500],[10,350],[300,200],[500,200],[800,350],[800,500]], np.int32)
##    processed_img = roi(processed_img,[vertices])
##    ## EDGES
##    lines = cv2.HoughLinesP(processed_img, 1, np.pi/180, 180, np.array([]), 100, 5)
##    m1 = 0
##    m2 = 0
##    try:
##        l1, l2, m1, m2 = draw_lanes(original_image, lines)
##        cv2.line(original_image, (l1[0], l1[1]),(l1[2], l1[3]), [0,255,0], 30)
##        cv2.line(original_image, (l2[0], l2[1]),(l2[2], l2[3]), [0,255,0], 30)
##    except Exception as e:
##        print(str(e))
##        pass
##    try:
##        for coords in lines:
##            coords = coords[0]
##            try:
##                cv2.line(processed_img,(coords[0], coords[1]), (coords[2], coords[3]), [255,0,0], 3)
##
##            except Exception as e:
##                print(str(e))
##    except Exception as e:
##            pass
##        
##    return processed_img,original_image, m1, m2
##
##def straight():
##    PressKey(W)
##    ReleaseKey(A)
##    ReleaseKey(D)
##
##def left():
##    PressKey(A)
##    PressKey(W)
##    ReleaseKey(D)
##    
##def right():
##    PressKey(D)
##    PressKey(W)
##    ReleaseKey(A)
##
##def slow():
##    ReleaseKey(W)
##    ReleaseKey(A)
##    ReleaseKey(D)

model = alexnet(WIDTH, HEIGHT, LR)
model.load(MODEL_NAME)

##Check data

'''train_data = np.load('AI_training_data.npy')
for data in train_data:
    img = data[0]
    choice = data[1]
    cv2.imshow('test',img)
    print(choice)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break'''

##file_name = 'AI_training_data.npy'
##
##if os.path.isfile(file_name):
##    print('File exists, loading previous data')
##    training_data = list(np.load(file_name, None, 1))
##else:
##    print("File does not exist, starting fresh")
##    training_data = []

for i in list(range(10))[::-1]:
    print(i+1)
    time.sleep(1)
def main():
    last_time = time.time()
    paused = False
    while True:

        if not paused:
            '''screen = grab_screen(region=(0,40,800,600))'''
            screen = np.array(ImageGrab.grab(bbox=(0,40,800,600)))
            screen = cv2.cvtColor(screen , cv2.COLOR_BGR2GRAY)
            screen = cv2.resize(screen, (80, 60))
            cv2.imshow('',screen)

##    TRACK KEY I.E KEYLOGGER
##            keys = key_check()
##            output = keys_to_output(keys)
##            training_data.append([screen,output])
##            new_screen,original_image, m1, m2 = process_img(screen)
            print('Loop took {} seconds'.format(time.time()-last_time))
            last_time = time.time()

            prediction = model.predict([screen.reshape(1,WIDTH,HEIGHT,1)[0]])
            moves = list(np.around(prediction))
            print(moves, prediction)

            if moves == [1,0,0]:
                PressKey(A)
                PressKey(W)
                ReleaseKey(D)
                print("LEFT")
            elif moves == [0,1,0]:
                PressKey(W)
                ReleaseKey(A)
                ReleaseKey(D)
                print("STRAIGHT")
            elif moves == [0,0,1]:
                PressKey(D)
                PressKey(W)
                ReleaseKey(A)
                print("RIGHT")

        keys = key_check()

        if 'T' in keys:
            if paused:
                paused = False
                time.sleep(1)
            else:
                paused = True
                ReleaseKey(A)
                ReleaseKey(W)
                ReleaseKey(D)
                time.sleep(1)
##COLLECTION OF DATA        
##        print(len(training_data))
##        if len(training_data) % 2000 == 0:
##            print("Data_Saved")
##            np.save(file_name, training_data)
##EDGE DETECTION WITH LANE DETECTION
##        cv2.imshow('Edge_Detection',new_screen)
##        cv2.imshow('Lane_Detection',cv2.cvtColor(original_image,cv2.COLOR_BGR2RGB))

##SLOPE DETECTION CONTROLS        
##        if m1 < 0 and m2 < 0 or m1 < 0:
##            right()
##        elif m1 > 0 and m2 > 0 or m2 > 0:
##            left()
##        else:
##            straight()
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
main()



