import csv
import sklearn
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks_cwt

from scipy.io.wavfile import read


from collections import deque
from math import pi, log
import pylab
import pdb
from scipy import fft, ifft
from scipy import signal
from scipy.optimize import curve_fit

import wave  
import time  
import sys
from scipy.fftpack import rfft
from sklearn.preprocessing import normalize

import scipy.io.wavfile as wav
from numpy.lib import stride_tricks

from scipy.interpolate import interp1d

max_slices = 10

def stfft(samples, samplerate, segment_size, roll=False):
    if roll:
        ims = np.empty((0,samplerate), dtype='float64')
        for i in xrange(0, len(samples) - samplerate, segment_size):
            ims = np.vstack((ims, rfft(samples[i:i+samplerate], samplerate)))
    else:
        ims = np.array([rfft(elem, samplerate) for elem in 
                np.array_split(samples, int(len(samples) / segment_size) + 1)],
                dtype="float64")
    #sshow, freq = stfft.logscale_spec(s, factor=1.0, sr=samplerate)

    ims = 20.*np.log10(np.abs(ims)/10e-6)

    ims[np.logical_not(np.isfinite(ims))] = 0
    #sample = normalize(ims, axis=1)
    #sample_chunked = [np.max(elem, axis=0) for elem in np.array_split(sample, int(len(sample) / segment_size) + 1)]
    return ims

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = signal.lfilter(b, a, data)
    return y

def maxfilter(samples, filter_size):
    vals = deque()
    max_samples = np.empty(len(samples))
    for i in xrange(len(samples)):
        vals.append(samples[i])
        if len(vals) > filter_size:
            vals.popleft()
        max_samples[i] = max(vals)
    return max_samples

def modefilter(samples, filter_size):
    vals = deque()
    max_samples = np.empty(len(samples))
    for i in xrange(len(samples)):
        vals.append(samples[i])
        if len(vals) > filter_size:
            vals.popleft()
        max_samples[i] = max(set(vals), key=vals.count)
    return max_samples

def peakfind(peak_samples, data_samples, time, samplerate):
    print "Finding Peaks..."
    samples = maxfilter(peak_samples, 100)
    indexes = signal.find_peaks_cwt(samples, np.arange(30, 100, 10))
    
    print "%d Peaks detected." % len(indexes)
    #adjusting for times
    basetime = int(time[0])

    for i in xrange(len(time)):
        time[i] = int(time[i]) - basetime

    translated_indexes = [int(time[idx] / 1000.0 * samplerate) for idx in indexes]
    #filteredidx = peakfilter(translated_indexes, data_samples, samplerate)
    return np.array(translated_indexes) #filteredidx

def peakfilter(peak_times, data_samples, samplerate):
    print "Filtering peaks..."
    if len(peak_times) < 1:
        return []
    prev_fft = maxfilter(rfft(data_samples[peak_times[0] 
                            : peak_times[1]], samplerate), 100)
    filtered_peaks = []
    ffts = []
    for i in xrange(len(peak_times) - 1):
        cur_fft = maxfilter(rfft(data_samples[peak_times[i] 
                            : peak_times[i+1]], samplerate), 100)
        ffts.append(np.sum(np.abs(cur_fft - prev_fft) ** 2))
        if np.sum(np.abs(cur_fft - prev_fft) ** 2) > 5e15:
            filtered_peaks.append(peak_times[i])
        prev_fft = cur_fft
    cur_fft = rfft(data_samples[peak_times[i] :], samplerate)
    if np.sum(np.abs(cur_fft - prev_fft) ** 2) > 5e15:
        filtered_peaks.append(peak_times[i])

    print "Removed %d peaks." % (len(peak_times) - len(filtered_peaks))
    return filtered_peaks

def splice(data, splittimes, slice_length):
    return_array = []
    prev_time = 0
    for i in xrange(len(splittimes)):
        if int(splittimes[i]) - prev_time > max_slices * slice_length:
            return_array.append(data[prev_time:prev_time+slice_length*max_slices])
        elif int(splittimes[i]) - prev_time > slice_length:
            return_array.append(data[prev_time : int(splittimes[i])])
        prev_time = int(splittimes[i])

    if splittimes[-1] - prev_time > max_slices * slice_length:
        return_array.append(data[prev_time:prev_time+slice_length*max_slices])
    elif splittimes[-1] - prev_time > slice_length:
        return_array.append(data[prev_time:])
    return return_array
def process_mag(magfile, truthfile="", peakfile="", use_peaks=True, accelfile=""):
    with open(magfile, 'rU') as f:
        reader = csv.reader(f)
        mag_data = np.array(list(reader), dtype='float64')
    x = mag_data[:, 3] - min(mag_data[:,3])
    fun0 = interp1d(x, mag_data[:,0] - np.mean(mag_data[:,0]))
    fun1 = interp1d(x, mag_data[:,1] - np.mean(mag_data[:,1]))
    fun2 = interp1d(x, mag_data[:,2] - np.mean(mag_data[:,2]))
    xnew = np.arange(min(x), max(x), 10)
    if use_peaks:
        if peakfile == "":
            with open(accelfile, 'rU') as f:
                csv_f = csv.reader(f)
                dataraw = np.array(list(csv_f), dtype='float64')
                data = normalize(dataraw, axis=0)
                accel_data = data[:,0] ** 2 + data[:,1] ** 2 + data[:,2] ** 2
                accel_time = dataraw[:,3]
            peaktimes = peakfind(accel_data, fun0(xnew), accel_time, 100)
        else:
            with open(peakfile, 'rU') as f:
                csv_f = csv.reader(f)
                peaktimes = np.array(list(csv_f), dtype='float64') / 10
        mag0_array = splice(fun0(xnew), peaktimes, 100);
        mag1_array = splice(fun1(xnew), peaktimes, 100);
        mag2_array = splice(fun2(xnew), peaktimes, 100);
    else:
        mag0_array = [fun0(xnew)]
        mag1_array = [fun1(xnew)]
        mag2_array = [fun2(xnew)]

    prep_array = np.empty((0,26*3))
    if truthfile != "":
        with open(truthfile, 'rU') as f:
            csv_f = csv.reader(f)
            truth_array = np.array(list(csv_f), dtype='float64')
        truth_label = []
    print "Preparing data..."
    arr0 = []
    arr1 = []
    arr2 = []
    arr3 = []
    for i in xrange(len(mag0_array)):
        if truthfile != '' and truth_array[i][0] > 500:
            continue
        if use_peaks:
            """mag0_prep = stfft(mag0_array[i], 100, 10000, roll=False)
            mag1_prep = stfft(mag1_array[i], 100, 10000, roll=False)
            mag2_prep = stfft(mag2_array[i], 100, 10000, roll=False)"""
        else:
            mag0_prep = stfft(mag0_array[i], 100, 100, roll=False)
            mag1_prep = stfft(mag1_array[i], 100, 100, roll=False)
            mag2_prep = stfft(mag2_array[i], 100, 100, roll=False)
        _, mag0_prep = signal.periodogram(mag0_array[i], nfft=50)
        _, mag1_prep = signal.periodogram(mag1_array[i], nfft=50)
        _, mag2_prep = signal.periodogram(mag2_array[i], nfft=50)
        #mag_prep = np.concatenate((mag0_prep, mag1_prep, mag2_prep), axis=1)
        if (len(mag0_prep) < 1):
            continue
        mag_prep = np.concatenate((mag0_prep, mag1_prep, mag2_prep), axis=0)

        prep_array = np.vstack((prep_array, mag_prep))
        #prep_array = np.concatenate((prep_array, mag_prep))
        if truthfile != "":
            if truth_array[i][0] < 91:
                truth_label.extend([0] * mag_prep.shape[0])
                arr0.append(mag_prep)
            elif truth_array[i][0] < 181:
                truth_label.extend([1] * mag_prep.shape[0])
                arr1.append(mag_prep)
            elif truth_array[i][0] < 271:
                truth_label.extend([2] * mag_prep.shape[0])
                arr2.append(mag_prep)
            elif truth_array[i][0] < 361:
                truth_label.extend([3] * mag_prep.shape[0])
                arr3.append(mag_prep)
    arr0.extend(arr1)
    arr0.extend(arr2)
    arr0.extend(arr3)
    if truthfile != "":
        return prep_array, np.array(truth_label)
    elif use_peaks and accelfile:
        return prep_array, peaktimes * 10
    else:
        return prep_array


def preprocess(wavfile, accelfile, truthfile='', 
               peakfile='', segment_size=44100, use_peaks=True):
    samplerate, wav_samples = read(wavfile)
    wav_data = (wav_samples[:,0] + wav_samples[:,1])/2
    
    if use_peaks:
        if peakfile == "":
            with open(accelfile, 'rU') as f:
                csv_f = csv.reader(f)
                dataraw = np.array(list(csv_f), dtype='float64')
                data = normalize(dataraw, axis=0)
                accel_data = data[:,0] ** 2 + data[:,1] ** 2 + data[:,2] ** 2
                accel_time = dataraw[:,3]
            peaktimes = peakfind(accel_data, wav_data, accel_time, samplerate)
        else:
            with open(peakfile, 'rU') as f:
                # TODO: Change to list
                csv_f = csv.reader(f)
                peaktimes = np.array(list(csv_f), dtype='float64') / 1000 * samplerate

        wav_array = splice(wav_data, peaktimes, 4410);
    else:
        wav_array = [wav_data]
    prep_array = np.empty((0,10000))
    
    if truthfile != "":
        with open(truthfile, 'rU') as f:
            csv_f = csv.reader(f)
            truth_array = np.array(list(csv_f), dtype='float64')
        truth_label = []
    print "Preparing data..."
    for i in xrange(len(wav_array)):
        if truthfile != '' and truth_array[i][0] > 500:
            continue
        wav_prep = stfft(wav_array[i], 10000, 4410)

        prep_array = np.concatenate((prep_array, wav_prep))
        if truthfile != "":
            truth_val = truth_array[i][0]
            if truth_val <= 90:
                truth_label.extend([truth_val] * wav_prep.shape[0])
            elif truth_val <= 180:
                truth_label.extend([180 - truth_val] * wav_prep.shape[0])
            elif truth_val <= 270:
                truth_label.extend([truth_val - 180] * wav_prep.shape[0])
            elif truth_val <= 360:
                truth_label.extend([360 - truth_val] * wav_prep.shape[0])
    if truthfile != "":
        return prep_array, np.array(truth_label)
    else:
        return prep_array

def main():
    #preprocess(wavfile, magfile, accelfile, trutharray)
    """
    par_dir = 'recordings/smartphone/'
    rec_num = '1430177546499'
    wavfile = par_dir + rec_num + '/' + rec_num + '.wav'
    magfile = par_dir + rec_num + '/' + rec_num + 'Mag.csv'
    accelfile = par_dir + rec_num + '/' + rec_num + 'Accel.csv'
    truthfile = par_dir + rec_num + '/' + rec_num + 'Truth.csv'
    """

    wavfile = 'training_data/0-180_combined_audio.wav'
    magfile = 'training_data/mag_data_merged.csv'
    accelfile = 'training_data/truth_values_merged.csv'

    data, truth = preprocess(wavfile, magfile, accelfile)

if __name__ == '__main__':
    main()
