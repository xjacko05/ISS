import sys
import numpy as np
from matplotlib import pyplot as plt
from scipy import signal
from scipy.io import wavfile

def dft(frame):
    dft_ret = np.ndarray((1024), dtype="complex")
    for k in range (0, 1024):
        dft_ret[k] = 0j
        for n in range (0, 1024):
            dft_ret[k] += frame[n]*np.exp(-1j*(2*np.pi/1024)*k*n)
    return dft_ret

def preprocess(subject):
    tmp = subject - np.mean(subject)
    if np.abs(subject.max()) > np.abs(subject.min()):
        tmp = tmp / np.abs(subject.max())
    else:
        tmp = tmp / np.abs(subject.min())

    res = np.ndarray((int(((len(subject) / 1024) * 2) - 1), 1024))
    for i in range (0, int(((len(subject) / 1024) * 2) - 1)):
        for j in range (0, 1024):
            res[i][j] = tmp[i*512+j]
    return res

def generate_spectro(frames):
    audio_frames_dft = np.ndarray((len(frames), 1024), dtype="complex")
    for i in range(0, len(frames)):
        audio_frames_dft[i] = np.fft.fft(frames[i], 1024)

    ret = np.ndarray((len(audio_frames_dft), 512))
    for i in range(0, len(audio_frames_dft)):
        ret[i] = 10 * np.log10(np.abs(audio_frames_dft[i][0:512])**2)
    ret = np.transpose(ret)
    return ret

show = False

if 's' in sys.argv:
    show = True


#task 1
fs, audio = wavfile.read('../audio/xjacko05.wav')
print("Signal min value is:\t", audio.min())
print("Signal max value is:\t", audio.max())

plt_audio_vanilla = plt.figure(1, figsize=(7,4))
plt.title('xjacko05.wav')
plt.ylabel('Signal value')
plt.xlabel('time[s]')
plt.plot(audio)
plt.xticks([0,8004,16008,24012,32016,40020], [0,0.5,1,1.5,2,2.5])
if show:
    plt.show()

#task 2
"""
tmp = audio - np.mean(audio)
if np.abs(audio.max()) > np.abs(audio.min()):
    tmp = tmp / np.abs(audio.max())
else:
    tmp = tmp / np.abs(audio.min())

audio_frames = np.ndarray((int(((len(audio) / 1024) * 2) - 1), 1024))
for i in range (0, int(((len(audio) / 1024) * 2) - 1)):
    for j in range (0, 1024):
        audio_frames[i][j] = tmp[i*512+j]
"""
audio_frames = preprocess(audio)

plt_audio_frame = plt.figure(2, figsize=(7,4))
plt.title('Frame 15')
plt.ylabel('Signal value')
plt.xlabel('time[s]')
plt.plot(audio_frames[15])
plt.xticks([0,127,255,383,511,639,767,895,1023], [0.4798,0.4878,0.4958,0.5037,0.5117,0.5197,0.5277,0.53570,0.5437])
if show:
    plt.show()

#task3
audio_frame_dft = dft(audio_frames[15])

plt_audio_frame_dft = plt.figure(3, figsize=(7,4))
plt.title('Frame 15 DFT - my implementation')
plt.ylabel('Coefficient value')
plt.xlabel('frequency[Hz]')
plt.plot(np.abs(audio_frame_dft[:512].real))
plt.xticks([0,63,127,191,255,319,383,447,511],[0,1000,2000,3000,4000,5000,6000,7000,8000])
if show:
    plt.show()


plt_audio_frame_fft = plt.figure(4, figsize=(7,4))
plt.title('Frame 15 DFT - numpy.fft.fft')
plt.ylabel('Coefficient value')
plt.xlabel('frequency[Hz]')
plt.plot(np.abs(np.fft.fft(audio_frames[15])[:512].real))
plt.xticks([0,63,127,191,255,319,383,447,511],[0,1000,2000,3000,4000,5000,6000,7000,8000])
if show:
    plt.show()

#task4
"""
audio_frames_dft = np.ndarray((len(audio_frames), 1024), dtype="complex")
for i in range(0, len(audio_frames)):
    audio_frames_dft[i] = np.fft.fft(audio_frames[i], 1024)

audio_spectrogram = np.ndarray((len(audio_frames_dft), 512))
for i in range(0, len(audio_frames_dft)):
    audio_spectrogram[i] = 10 * np.log10(np.abs(audio_frames_dft[i][0:512])**2)
audio_spectrogram = np.transpose(audio_spectrogram)
"""
audio_spectrogram = generate_spectro(audio_frames)

plt_audio_spectrogram = plt.figure(5, figsize=(7,4))
plt.title('Spectrogram of original audio')
plt.xlabel('time[s]')
plt.ylabel('frequency[Hz]')
bar = plt.imshow(audio_spectrogram, extent=[0,2.45,0,8000], origin='lower', aspect='auto')
plt.colorbar(bar)
if show:
    plt.show()

#task6
audio_4cos = np.ndarray((len(audio)))
for n in range (0, len(audio)):
    audio_4cos[n] = np.cos(2*np.pi*(665/16000)*n)
    audio_4cos[n] += np.cos(2*np.pi*(1330/16000)*n)
    audio_4cos[n] += np.cos(2*np.pi*(1995/16000)*n)
    audio_4cos[n] += np.cos(2*np.pi*(2660/16000)*n)

wavfile.write('../audio/4cos.wav', 16000, audio_4cos)

plt_4cos_spectrogram = plt.figure(6, figsize=(7,4))
plt.title('Spectrogram of interference')
plt.xlabel('time[s]')
plt.ylabel('frequency[Hz]')
bar = plt.imshow(generate_spectro(preprocess(audio_4cos)), extent=[0,2.45,0,8000], origin='lower', aspect='auto')
plt.colorbar(bar)
plt.show()

print('FINISHED')