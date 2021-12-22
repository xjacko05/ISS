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

def butter_bandpass(lowcut, highcut, fs, ripple, atten):
    nyq = fs / 2
    low = lowcut / nyq
    high = highcut / nyq
    lowish = (lowcut - 50) / nyq
    highish = (highcut + 50) / nyq
    order, wn = signal.buttord([lowish, highish], [low, high], ripple, atten)
    #print([low, high])
    #print([lowish, highish])
    #print(wn)
    #print(order)
    b, a = signal.butter(order, wn, btype='bandstop')
    print("Filter coefficients for frequency ", (lowcut+highcut)/2, "\tare:\n", b, "\n", a, "\n\n")
    #print("Filter coefficients for frequency ", (lowcut+highcut)/2, "\tare:\n", b/a, "\n\n")
    #w, h = signal.freqz(b , a)
    #templot = plt.figure(11, figsize=(7,4))
    #plt.plot((16000 * 0.5 / np.pi) *w, abs(h))
    #plt.show()
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, ripple, atten):
    b, a = butter_bandpass(lowcut, highcut, fs, ripple, atten)
    y = signal.lfilter(b, a, data)
    return y


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
if show:
    plt.show()

#task7
filter_1 = butter_bandpass(650, 680, 16000, 3, 40)
filter_2 = butter_bandpass(1315, 1345, 16000, 3, 40)
filter_3 = butter_bandpass(1980, 2010, 16000, 3, 40)
filter_4 = butter_bandpass(2645, 2675, 16000, 3, 40)



fs, audio_vanilla = wavfile.read('../audio/xjacko05.wav')
#audio_clean = butter_bandpass_filter(audio_vanilla, 650, 680, 16000, 3, 40)
#audio_clean = butter_bandpass_filter(audio_clean, 1315, 1345, 16000, 3, 40)
#audio_clean = butter_bandpass_filter(audio_clean, 1980, 2010, 16000, 3, 40)
#audio_clean = butter_bandpass_filter(audio_clean, 2645, 2675, 16000, 3, 40)
audio_clean = signal.lfilter(filter_1[0], filter_1[1], audio_vanilla)
audio_clean = signal.lfilter(filter_2[0], filter_2[1], audio_clean)
audio_clean = signal.lfilter(filter_3[0], filter_3[1], audio_clean)
audio_clean = signal.lfilter(filter_4[0], filter_4[1], audio_clean)
wavfile.write('../audio/clean_aaa.wav', 16000, audio_clean)

impulse_input = np.ndarray((512))
for i in range (0,512):
    impulse_input[i] = 0
impulse_input[0] = 1

plt_filter_1 = plt.figure(7, figsize=(7,4))
plt.title('Filter 1 (655 Hz) impulse response')
plt.plot(signal.lfilter(filter_1[0], filter_1[1], impulse_input))
if show:
    plt.show()

plt_filter_2 = plt.figure(8, figsize=(7,4))
plt.title('Filter 2 (1330 Hz) impulse response')
plt.plot(signal.lfilter(filter_2[0], filter_2[1], impulse_input))
if show:
    plt.show()

plt_filter_3 = plt.figure(9, figsize=(7,4))
plt.title('Filter 3 (1995 Hz) impulse response')
plt.plot(signal.lfilter(filter_3[0], filter_3[1], impulse_input))
if show:
    plt.show()

plt_filter_4 = plt.figure(10, figsize=(7,4))
plt.title('Filter 4 (2660 Hz) impulse response')
plt.plot(signal.lfilter(filter_4[0], filter_4[1], impulse_input))
if show:
    plt.show()

#task8
zero_pole_1 = signal.tf2zpk(filter_1[0], filter_1[1])

plt.figure(99, figsize=(4,3.5))

# jednotkova kruznice
ang = np.linspace(0, 2*np.pi,100)
plt.plot(np.cos(ang), np.sin(ang))

# nuly, poly
plt.scatter(np.real(zero_pole_1[0]), np.imag(zero_pole_1[0]), marker='o', facecolors='none', edgecolors='r', label='nuly')
plt.scatter(np.real(zero_pole_1[1]), np.imag(zero_pole_1[1]), marker='x', color='g', label='póly')

plt.gca().set_xlabel('Realná složka $\mathbb{R}\{$z$\}$')
plt.gca().set_ylabel('Imaginarní složka $\mathbb{I}\{$z$\}$')

plt.grid(alpha=0.5, linestyle='--')
plt.legend(loc='upper right')

plt.tight_layout()







#task9
freq_char_1 = signal.freqz(filter_1[0], filter_1[1])

plt_freq_char_1_modul = plt.figure(11, figsize=(7,4))
plt.plot(freq_char_1[0] / 2 / np.pi * fs, np.abs(freq_char_1[1]))
plt.title('Module of Filter 1 (665 Hz) frequency characteristics $|H(e^{j\omega})|$')
plt.xlabel('frequency[Hz]')
if show:
    plt.show()

plt_freq_char_1_argument = plt.figure(12, figsize=(7,4))
plt.plot(freq_char_1[0] / 2 / np.pi * fs, np.angle(freq_char_1[1]))
plt.title('Argument of Filter 1 (665 Hz) frequency characteristics $\mathrm{arg}\ H(e^{j\omega})$')
plt.xlabel('frequency[Hz]')
if show:
    plt.show()

freq_char_2 = signal.freqz(filter_2[0], filter_2[1])

plt_freq_char_2_modul = plt.figure(13, figsize=(7,4))
plt.plot(freq_char_2[0] / 2 / np.pi * fs, np.abs(freq_char_2[1]))
plt.title('Module of Filter 2 (1330 Hz) frequency characteristics $|H(e^{j\omega})|$')
plt.xlabel('frequency[Hz]')
if show:
    plt.show()

plt_freq_char_2_argument = plt.figure(14, figsize=(7,4))
plt.plot(freq_char_2[0] / 2 / np.pi * fs, np.angle(freq_char_2[1]))
plt.title('Argument of Filter 2 (1330 Hz) frequency characteristics $\mathrm{arg}\ H(e^{j\omega})$')
plt.xlabel('frequency[Hz]')
if show:
    plt.show()

freq_char_3 = signal.freqz(filter_3[0], filter_3[1])

plt_freq_char_3_modul = plt.figure(15, figsize=(7,4))
plt.plot(freq_char_3[0] / 2 / np.pi * fs, np.abs(freq_char_3[1]))
plt.title('Module of Filter 3 (1995 Hz) frequency characteristics $|H(e^{j\omega})|$')
plt.xlabel('frequency[Hz]')
if show:
    plt.show()

plt_freq_char_3_argument = plt.figure(16, figsize=(7,4))
plt.plot(freq_char_3[0] / 2 / np.pi * fs, np.angle(freq_char_3[1]))
plt.title('Argument of Filter 3 (1995 Hz) frequency characteristics $\mathrm{arg}\ H(e^{j\omega})$')
plt.xlabel('frequency[Hz]')
if show:
    plt.show()

freq_char_4 = signal.freqz(filter_4[0], filter_4[1])

plt_freq_char_4_modul = plt.figure(17, figsize=(7,4))
plt.plot(freq_char_4[0] / 2 / np.pi * fs, np.abs(freq_char_4[1]))
plt.title('Module of Filter 4 (2660 Hz) frequency characteristics $|H(e^{j\omega})|$')
plt.xlabel('frequency[Hz]')
if show:
    plt.show()

plt_freq_char_4_argument = plt.figure(18, figsize=(7,4))
plt.plot(freq_char_4[0] / 2 / np.pi * fs, np.angle(freq_char_4[1]))
plt.title('Argument of Filter 4 (2660 Hz) frequency characteristics $\mathrm{arg}\ H(e^{j\omega})$')
plt.xlabel('frequency[Hz]')
plt.show()

"""
_, ax = plt.subplots(11, 12, figsize=(8,3))

ax[0].plot(freq_char_1[0] / 2 / np.pi * fs, np.abs(freq_char_1[1]))
ax[0].set_xlabel('Frekvence [Hz]')
ax[0].set_title('Modul frekvenční charakteristiky $|H(e^{j\omega})|$')

ax[1].plot(freq_char_1[0] / 2 / np.pi * fs, np.angle(freq_char_1[1]))
ax[1].set_xlabel('Frekvence [Hz]')
ax[1].set_title('Argument frekvenční charakteristiky $\mathrm{arg}\ H(e^{j\omega})$')

for ax1 in ax:
    ax1.grid(alpha=0.5, linestyle='--')

plt.tight_layout()
"""

print('FINISHED')