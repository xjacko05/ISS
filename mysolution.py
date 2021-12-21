import sys
import numpy as np
from matplotlib import pyplot as plt
from scipy import signal
from scipy.io import wavfile

def dft(frame):
    dft_ret = np.ndarray((1024), dtype="complex")
    for k in range (0, 1023):
        dft_ret[k] = 0j
        for n in range (0, 1023):
            dft_ret[k] += frame[n]*np.exp(-1j*(2*np.pi/1024)*k*n)
    return dft_ret

show = False

if 's' in sys.argv:
    show = True


#task 1
fs, audio = wavfile.read('xjacko05.wav')
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
tmp = audio - np.mean(audio)
if np.abs(audio.max()) > np.abs(audio.min()):
    tmp = tmp / np.abs(audio.max())
else:
    tmp = tmp / np.abs(audio.min())

audio_frames = np.ndarray((int(((len(audio) / 1024) * 2) - 1), 1024))
for i in range (0, int(((len(audio) / 1024) * 2) - 1)):
    for j in range (0, 1023):
        audio_frames[i][j] = tmp[i*512+j]

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
plt.xticks([0,127,255,383,511],[0,2000,4000,6000,8000])
if show:
    plt.show()


plt_audio_frame_fft = plt.figure(4, figsize=(7,4))
plt.title('Frame 15 DFT - numpy.fft.fft')
plt.ylabel('Coefficient value')
plt.xlabel('frequency[Hz]')
plt.plot(np.abs(np.fft.fft(audio_frames[15])[:512].real))
plt.xticks([0,127,255,383,511],[0,2000,4000,6000,8000])
if show:
    plt.show()

#task4
plot6 = plt.figure(6)
bar = plt.imshow(spectogram_maskon, extent=[0,1,0,8000], origin='lower', aspect='auto')
plt.colorbar(bar)
plt.xlabel('t[s]')
plt.ylabel('freq')
plt.title('Spectogram mask on')

#for k in range (0, int(((len(audio) / 1024) * 2) - 1)):

    #plt.clf()
    #plt.title(k)
    #plt.plot(audio_frames[k])
    #if show:
        #plt.show()
    #plt.clf()



print("FINISHED")