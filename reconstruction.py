from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from scipy.io.wavfile import read
import wave
import numpy as np
import scipy.signal as signal
import pdb
import wave
import struct
import pickle
import math
import preprocessing

def main():
    with open("mag.model") as f:
        magmodel = pickle.load(f)
    with open("wav.model") as f:
        wavmodel = pickle.load(f)
    preds = []

    wavfile = '/Users/sunny/Desktop/trainingdata5816/1462752256592.wav'
    magfile = '/Users/sunny/Desktop/trainingdata5816/1462752256592Mag.csv'
    accelfile = '/Users/sunny/Desktop/trainingdata5816/1462752256592Accel.csv'
    
    wavfile = '/Users/sunny/Desktop/trainingdata5816/1462752609208.wav'
    magfile = '/Users/sunny/Desktop/trainingdata5816/1462752609208Mag.csv'
    accelfile = '/Users/sunny/Desktop/trainingdata5816/1462752609208Accel.csv'
    
    magdata = preprocessing.process_mag(magfile, accelfile=accelfile, use_peaks=False)
    magpred = magmodel.predict(magdata)
    wavdata = preprocessing.preprocess(wavfile, accelfile, use_peaks=False)
    data_tot = []
    wavpred = wavmodel.predict(wavdata)

    preds_smoothed = preprocessing.modefilter(wavpred, 21)
    cur_x = 0
    cur_y = 0
    x_list = []
    y_list = []
    step_size = 1.0
    pdb.set_trace()
    for i in xrange(0, len(preds_smoothed)):
        pred = preds_smoothed[i]
        if i/10 >= len(magpred):
            break
        mag = magpred[i/10]
        if mag == 0:
            cur_x += step_size * math.cos(math.pi * pred / 180.0)
            cur_y += step_size * math.sin(math.pi * pred / 180.0)
        if mag == 1:
            cur_x += step_size * math.cos(math.pi * (180 - pred) / 180.0)
            cur_y += step_size * math.sin(math.pi * (180 - pred) / 180.0)
        if mag == 2:
            cur_x += step_size * math.cos(math.pi * (pred + 180) / 180.0)
            cur_y += step_size * math.sin(math.pi * (pred + 180) / 180.0)
        if mag == 3:
            cur_x += step_size * math.cos(math.pi * (360 - pred) / 180.0)
            cur_y += step_size * math.sin(math.pi * (360 - pred) / 180.0)
        x_list.append(cur_x)
        y_list.append(cur_y)
    plt.plot(x_list, y_list)
    plt.show()


if __name__ == '__main__':
    main()