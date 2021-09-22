from numpy.lib import stride_tricks
import scipy.io.wavfile as wav
from matplotlib import pyplot as plot
import numpy as np
import speech_recognition as sr

def Short_Time_Fourier_Transforms(signal, Frame_Size, step_size=0.5, spectrograph=np.hanning):
    window = spectrograph(Frame_Size)
    # to decrease the frame size
    hopSize = int(Frame_Size - np.floor(step_size * Frame_Size))
    # zeros at beginning (thus center of 1st spectrograph should be for sample nr. 0)
    samples = np.append(np.zeros(int(np.floor(Frame_Size / 2.0))), signal)
    # columns for windowing
    columns = np.ceil((len(samples) - Frame_Size) / float(hopSize)) + 1
    # zeros at end (thus samples can be fully covered by frames)
    samples = np.append(samples, np.zeros(Frame_Size))
    # Creating a view into the array with the given shape and strides.
    frames = stride_tricks.as_strided(samples, shape=(int(columns), Frame_Size),
                                      strides=(samples.strides[0] * hopSize, samples.strides[0])).copy()
    frames *= window
    # returning a one-dimensional discrete Fourier Transform for the real input given for number of time intervals
    print(len(frames))
    return np.fft.rfft(frames)

def logscale_spectrogram(spec, sr=44100, factor=20.):
    Time_bins, Frequency_bins = np.shape(spec)
    # Return evenly spaced numbers over a specified interval for freqeuncy
    scale = np.linspace(0, 1, Frequency_bins) ** factor
    scale *= (Frequency_bins - 1) / max(scale)
    # find unique elements in the array and to round them
    scale = np.unique(np.round(scale))
    # create spectrogram with new freq bins
    new_spectrogram = np.complex128(np.zeros([Time_bins, len(scale)]))
    for i in range(0, len(scale)):
        if i == len(scale) - 1:
            new_spectrogram[:, i] = np.sum(spec[:, int(scale[i]):], axis=1)
        else:
            new_spectrogram[:, i] = np.sum(spec[:, int(scale[i]):int(scale[i + 1])], axis=1)
    # list center freq of bins
    All_frequency = np.abs(np.fft.fftfreq(Frequency_bins * 2, 1. / sr)[:Frequency_bins + 1])
    Frequencies = []
    for i in range(0, len(scale)):
        if i == len(scale) - 1:
            Frequencies += [np.mean(All_frequency[int(scale[i]):])]
        else:
            Frequencies += [np.mean(All_frequency[int(scale[i]):int(scale[i + 1])])]
    print(len(Frequencies))
    return new_spectrogram, Frequencies

def Plot_Short_Time_Fourier_Transforms(audiopath, Bin_Size=2 ** 10, plotpath=None, colormap="jet"):
    Sample_rate, samples = wav.read(audiopath)
    s = Short_Time_Fourier_Transforms(samples, Bin_Size)
    sshow, freq = logscale_spectrogram(s, factor=1.0, sr=Sample_rate)
    Amplitude2Power = 20. * np.log10(np.abs(sshow) / 10e-6)  # amplitude to decibel
    Time_bins, Frequency_bins = np.shape(Amplitude2Power)
    print("Time_bins: ", Time_bins)
    print("Frequency_bins: ", Frequency_bins)
    plot.figure(figsize=(15, 7.5))
    plot.imshow(np.transpose(Amplitude2Power), origin="lower", aspect="auto", cmap=colormap, interpolation="none")
    plot.colorbar()
    plot.xlabel("Time (seconds)")
    plot.ylabel("Frequency (hertz)")
    plot.xlim([0, Time_bins - 1])
    plot.ylim([0, Frequency_bins])
    x_axis = np.float32(np.linspace(0, Time_bins - 1, 5))
    plot.xticks(x_axis, ["%.02f" % l for l in ((x_axis * len(samples) / Time_bins) + (0.5 * Bin_Size)) / Sample_rate])
    y_axis = np.int16(np.round(np.linspace(0, Frequency_bins - 1, 10)))
    plot.yticks(y_axis, ["%.02f" % freq[i] for i in y_axis])
    if plotpath:
        plot.savefig(plotpath, bbox_inches="tight")
    else:
        plot.show()
    plot.clf()
    return Amplitude2Power

Amplitude2Power = Plot_Short_Time_Fourier_Transforms('D:\\voice samples\\hellop.wav')
r = sr.Recognizer()
demo = sr.AudioFile("D:\\voice samples\\hellop.wav")
with demo as source:
    audio = r.record(source)
print(r.recognize_google(audio))
