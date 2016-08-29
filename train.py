from scipy.fftpack import rfft
from scipy.io.wavfile import read
import os
import fnmatch
from sklearn import svm, decomposition, linear_model
from sklearn.preprocessing import normalize
from sklearn.pipeline import Pipeline
import numpy as np
import pdb
import pickle
import preprocessing
from numpy import amax
import matplotlib.pyplot as plt
new_model = True
def main():
    wavfile = '/Users/sunny/Desktop/trainingdata5816/1462755111406.wav'
    magfile = '/Users/sunny/Desktop/trainingdata5816/1462755111406Mag.csv'
    accelfile = '/Users/sunny/Desktop/trainingdata5816/1462755111406Accel.csv'
    truthfile='/Users/sunny/Desktop/trainingdata5816/angle_list_180to0degprint.csv'
    peakfile='/Users/sunny/Desktop/trainingdata5816/1462755111406peaks.csv'

    mag_data, mag_truth = preprocessing.process_mag(magfile, truthfile, peakfile)
    wav_data, wav_truth = preprocessing.preprocess(wavfile, accelfile, truthfile, use_peaks=True)


    print "Begin Training..."
    if new_model:
        wav_learner = linear_model.LogisticRegression(solver='lbfgs', verbose=10, multi_class='multinomial', C=0.05)
        mag_learner = linear_model.LogisticRegression(solver='sag', verbose=10, C=0.05)
    else:
        with open("pipe.model", "rb") as f:
            pipe = pickle.load(f)
    mag_learner.fit(mag_data, mag_truth)
    print "Mag Score:"
    print mag_learner.score(mag_data, mag_truth)
   
    wav_learner.fit(wav_data, wav_truth)

    print "Wav Score:"
    print wav_learner.score(wav_data, wav_truth)


    f = open("wav.model", "wb")
    pickle.dump(wav_learner.sparsify(), f)

    
    f = open("mag.model", "wb")
    pickle.dump(mag_learner.sparsify(), f)



if __name__ == '__main__':
    main()