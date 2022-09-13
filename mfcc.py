import librosa as lbs, librosa.display as lbsdisplay
import matplotlib.pyplot as plt
import numpy as np
filepath = "/Users/nicholas/Desktop/Misc/Workspace/PythonWorkspace/musicProject/musicdata/genres_original/blues/blues.00000.wav"

signal, sr = lbs.load(filepath, sr=22050) # sr * T -> 22050 * 30s = >600000 values in signal

lbsdisplay.waveplot(signal, sr=sr)
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.show()


# fft = np.fft.fft(signal)
# get fourier transform
magnitude = np.abs(np.fft.fft(signal))
frequency = np.linspace(0, sr, len(magnitude))
leftFrequency = frequency[:int(len(frequency)/2)]
leftMagnitude = magnitude[:int(len(magnitude)/2)]

# plot power spectrum
plt.plot(leftFrequency, leftMagnitude)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude (dB)")
plt.show()

# get stft to get spectrogram
n_fft = 2048 # regular value for number of windows
hop_legnth = 512 # regular value for window jump

stft = lbs.core.stft(signal, n_fft=n_fft, hop_length=hop_legnth)
spectrogram = np.abs(stft)

# get log scale spectrogram to normalize
logSpec = lbs.amplitude_to_db(spectrogram)

lbsdisplay.specshow(logSpec, sr=sr, hop_length=hop_legnth)
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.colorbar()
plt.show()

# get mfccs
mfccs = lbs.feature.mfcc(signal, sr=sr, n_fft=n_fft, hop_length=hop_legnth, n_mfcc=13)
lbsdisplay.specshow(mfccs, sr=sr, hop_length=hop_legnth)
plt.xlabel("Time")
plt.ylabel("MFCC")
plt.colorbar()
plt.show()
