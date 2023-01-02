'''

Title: demodulation
Function: View a stream code data file. 

This function has a rolling window pearson cross-correlation. 

Author: Jean Rintoul
Date: 18.03.2022

'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq,fftshift,ifft,ifftshift
from scipy import conj
from scipy.signal import blackman
from scipy.signal import kaiserord, lfilter, firwin, freqz
from scipy import signal
from scipy.signal import butter, lfilter,iirfilter,sosfilt
from scipy.signal import fftconvolve
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from scipy.signal import find_peaks
from scipy import stats
import pandas as pd
import scipy.stats as st

def demodulate(signal,carrier_f):
    fc    = carrier_f
    phi   = 0 
    phi2  = 90 # this is for lock in like implementation. 
    A1    = 1.0 
    wc    = 2*np.pi*fc*t + phi
    wc2   = 2*np.pi*fc*t + phi2
    offset  = 0 #
    carrier_zero    = A1*np.exp(1j*(wc))+offset
    carrier_90      = A1*np.exp(1j*(wc2))+offset
    # multiply the carrier signal by the real signal. 
    multiplied_zero   = signal*carrier_zero
    multiplied_90     = signal*carrier_90
    demodulated_signal = multiplied_zero+multiplied_90
    return demodulated_signal 

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

def lowpassfilter(raw_signal,cutoff='50000'):
    #  Filter Design 
    nyq_rate = Fs / 2.0
    # # The desired width of the transition from pass to stop,
    # relative to the Nyquist rate.  
    width = 500.0/nyq_rate
    # The desired attenuation in the stop band, in dB.
    ripple_db = 60.0
    # Compute the order and Kaiser parameter for the FIR filter.
    Ntaps, beta = kaiserord(ripple_db, width)
    # The cutoff frequency of the filter.
    sos = iirfilter(17, [cutoff], rs=60, btype='lowpass',
                           analog=False, ftype='cheby2', fs=Fs,
                           output='sos')
    fsignal = sosfilt(sos, raw_signal)

    return fsignal

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx],idx    


Fs = 5e6
duration = 0.1
timestep = 1.0/Fs
N = int(Fs*duration)
print("expected no. samples:",N)
t = np.linspace(0, duration, N, endpoint=False)
t_orig = t 
xf = np.fft.fftfreq(N, d=timestep)[:N//2]
frequencies = xf[1:N//2]

rf_channel          = 0    
v_channel           = 4
ae_channel          = 6 
vmeasure_channel    = 5 
folder = 'ts_demod_evsae_2/'
filename = folder +'12.0_demod_8_stream.npy'

data = np.load(filename)
print (data.shape)
fft_vmfield = fft(10*data[vmeasure_channel])
fft_vm = fft(10*data[vmeasure_channel])

carrier_f               = 500000
filter_cutoff           = 20000
raw_demodulated_signal  = demodulate(data[ae_channel],carrier_f)
demodulated_signal      = lowpassfilter(raw_demodulated_signal,filter_cutoff)
raw_v_at_electrode      = lowpassfilter(10*data[vmeasure_channel],filter_cutoff)

# # remove the starting ramp and ending ramp. 
val,idx_start   = find_nearest(t,0.03)
val,idx_end     = find_nearest(t,0.08)
# clip the start and end areas away where the current signal is ramping. 
raw_v_at_electrode = raw_v_at_electrode[idx_start:idx_end]
demodulated_signal = demodulated_signal[idx_start:idx_end]
t_cut              = t[idx_start:idx_end]
# 
# Correctly conpute the lag. 
# 
x = raw_v_at_electrode
y = demodulated_signal
correlation = signal.correlate(x-np.mean(x), y - np.mean(y), mode="full")
lags = signal.correlation_lags(len(x), len(y), mode="full")
idx_lag = -lags[np.argmax(correlation)]
print ('lag is',idx_lag)
#  
if idx_lag > 0:
    raw_v_at_electrode = raw_v_at_electrode[:(len(demodulated_signal)-idx_lag)]
    t_cut              = t_cut[idx_lag:]
    demodulated_signal = demodulated_signal[idx_lag:]
else: 
    raw_v_at_electrode = raw_v_at_electrode[-idx_lag:]
    t_cut              = t_cut[-idx_lag:]
    demodulated_signal = demodulated_signal[:(len(demodulated_signal)+idx_lag)]
#  
demodulated_signal = demodulated_signal-np.mean(demodulated_signal)
raw_v_at_electrode = raw_v_at_electrode-np.mean(raw_v_at_electrode)

print ('lengths:',len(demodulated_signal),len(raw_v_at_electrode))

df = pd.DataFrame({'x': np.real(raw_v_at_electrode), 'y': np.real(demodulated_signal) })
window          = 12000
# window          = len(demodulated_signal)
rolling_corr    = df['x'].rolling(window).corr(df['y'])

print ('rolling corr max:',np.max(rolling_corr))  # 
result = np.nanmedian(rolling_corr)
print ('median corr: ',result)
max_index = np.argmax(rolling_corr) 
print ('max index',max_index,max_index+window)

print ('max length:',len(raw_v_at_electrode))

fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(2,1,1)
# plt.plot(t_cut,np.real(raw_v_at_electrode),'r')
# plt.plot(t_cut,np.real(demodulated_signal),'black')
# plt.axvline(x=t_cut[max_index],color='g')
# plt.axvline(x=t_cut[max_index+window] ,color='g')


plt.plot(np.real(raw_v_at_electrode),'r')
plt.plot(np.real(demodulated_signal),'black')
# plt.axvline(x=t_cut[max_index],color='g')
# plt.axvline(x=t_cut[max_index+window] ,color='g')

plt.legend(['raw v at electrode','demodulated signal'])
ax = fig.add_subplot(2,1,2)
plt.plot(t_cut,np.real(abs(rolling_corr)),'.b')
plt.legend(['abs(correlation) window:'+str(window)])
plt.show()


# # # #  Plots for paper. # #  #  #  #  
# These are the "Tableau Color Blind 10" colors as RGB.    
tableaucolorblind10 = [
(0, 107, 164), 
(255, 128, 14), 
(171, 171, 171), 
(89, 89, 89),    
(95, 158, 209), 
(200, 82, 0), 
(137, 137, 137), 
(162, 200, 236),  
(255, 188, 121), 
(207, 207, 207)]    
  
# Scale the RGB values to the [0, 1] range, which is the format matplotlib accepts.    
for i in range(len(tableaucolorblind10)):    
    r, g, b = tableaucolorblind10[i]    
    tableaucolorblind10[i] = (r / 255., g / 255., b / 255.) 
plt.rcParams['axes.linewidth'] = 2

start_pt = 0.03
end_pt   = 0.0325
fsize = 24 
save_title = 'ionic_input_signal'
fig = plt.figure(figsize=(7,7))
ax = fig.add_subplot(1,1,1)
plt.plot(t_cut,np.real(raw_v_at_electrode),color = tableaucolorblind10[3],linewidth=2.0 )

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
# plt.xticks([])
# plt.yticks([])
# plt.axis('off')
plt.xticks([0.03,0.0325],['0', '2.5'],fontsize=fsize)
# plt.xticks([0.03,0.0325],['0.03','0.0325'],fontsize=fsize)

plt.yticks([-0.1,0,0.1],['-0.1','0','0.1'],fontsize=fsize)
ax.set_xlim([start_pt,end_pt])
plt.savefig(save_title+".png",transparent=True, pad_inches=0,bbox_inches='tight')
plt.show()




save_title = 'demodulated_signal'
fig = plt.figure(figsize=(7,7))
ax = fig.add_subplot(1,1,1)
plt.plot(t_cut,np.real(demodulated_signal),color = tableaucolorblind10[3] ,linewidth=2.0)
ax.set_xlim([start_pt,end_pt])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
# plt.xticks([0.03,0.0325],['0.03','0.0325'],fontsize=fsize)
plt.xticks([0.03,0.0325],['0', '2.5'],fontsize=fsize)
plt.yticks([-0.1,0,0.1],['-0.1','0','0.1'],fontsize=fsize)
# plt.yticks([0],['0'],fontsize=fsize)
# plt.xticks([])
# plt.yticks([])
# plt.axis('off')
plt.savefig(save_title+".png",transparent=True, pad_inches=0,bbox_inches='tight')
plt.show()

d_lowcut  = 480000 # 504000
d_highcut = 520000 # 514000
sos_demod = signal.iirfilter(17, [d_lowcut, d_highcut], rs=60, btype='band',
                       analog=False, ftype='cheby2', fs=Fs,
                       output='sos')
ae_demod_input = signal.sosfilt(sos_demod, data[ae_channel])

save_title = 'heterodyned_signal'
fig = plt.figure(figsize=(7,7))
ax = fig.add_subplot(1,1,1)
plt.plot(t,ae_demod_input,color = tableaucolorblind10[3],linewidth=2.0 )
 
# plt.axis('off')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
# plt.xticks([])
# plt.yticks([])

plt.xticks([0.03,0.0325],['0', '2.5'],fontsize=fsize)
# plt.yticks([0],['0'],fontsize=fsize)
plt.yticks([-0.4,0,0.4],['-0.4','0','0.4'],fontsize=fsize)
ax.set_xlim([start_pt,end_pt])
plt.savefig(save_title+".png",transparent=True, pad_inches=0,bbox_inches='tight')
plt.show()

# # Now plot the FFT amplitude spectrum of each signal. 
t = t_cut 
N = len(t)
xf = np.fft.fftfreq(N, d=timestep)[:N//2]
frequencies = xf[1:N//2]

ionic_input_signal = np.real(raw_v_at_electrode)
recovered_signal   = np.real(demodulated_signal)
modulated_signal   = ae_demod_input

fft_ionic = fft(ionic_input_signal)
fft_ionic_input = np.abs(2.0/N * (fft_ionic))[1:N//2]



fft_recovered = fft(recovered_signal)
fft_recovered = np.abs(2.0/N * (fft_recovered))[1:N//2]

fsize = 16 

save_title = 'fft_ionic_signal'
fig = plt.figure(figsize=(6,3))
ax = fig.add_subplot(1,1,1)
plt.plot(frequencies/1000,fft_ionic_input,color = tableaucolorblind10[3],linewidth=2.0 )
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
# plt.xticks(fontsize=fsize)  
# plt.yticks([])
plt.yticks([0.0,0.03],fontsize=fsize)
ax.set_xlim([0,15])
# plt.xticks([0,8,15],['0','8','15'],fontsize=fsize)
plt.savefig(save_title+".png",transparent=True, pad_inches=0,bbox_inches='tight')
plt.show()


N = len(t_orig)
xf = np.fft.fftfreq(N, d=timestep)[:N//2]
frequenciesm = xf[1:N//2]
fft_modulated = fft(modulated_signal)
fft_modulated = np.abs(2.0/N * (fft_modulated))[1:N//2]

save_title = 'fft_modulated_signal'
fig = plt.figure(figsize=(6,3))
ax = fig.add_subplot(1,1,1)
plt.plot(frequenciesm/1000,fft_modulated,color = tableaucolorblind10[3],linewidth=2.0 )
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
# plt.xticks(fontsize=fsize)  
plt.yticks([0.0,0.06],fontsize=fsize)

plt.xticks([0,500],['0','500'],fontsize=fsize)
ax.set_xlim([0,550])
plt.savefig(save_title+".png",transparent=True, pad_inches=0,bbox_inches='tight')
plt.show()



save_title = 'fft_recovered_signal'
fig = plt.figure(figsize=(6,3))
ax = fig.add_subplot(1,1,1)
plt.plot(frequencies/1000,fft_recovered,color = tableaucolorblind10[3],linewidth=2.0 )
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
# plt.xticks(fontsize=fsize)  
plt.yticks([0.0,0.02],fontsize=fsize)
# plt.yticks([])
ax.set_xlim([0,15])
plt.xticks([0,8,15],['0','8','15'],fontsize=fsize)
plt.savefig(save_title+".png",transparent=True, pad_inches=0,bbox_inches='tight')
plt.show()