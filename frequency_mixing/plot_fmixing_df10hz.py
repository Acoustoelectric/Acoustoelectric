'''

Author: Jean Rintoul 
Date: 29/12/2022 

'''


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq,fftshift,ifft,ifftshift
from scipy.signal import blackman
from scipy.signal import kaiserord, lfilter, firwin, freqz
from scipy import signal
from scipy.signal import butter, lfilter
from scipy.signal import fftconvolve

#
# Create plots to show the difference frequency at 10Hz, when applied frequency is 499.99kHz
filename = '10_ati_stream_data_0.8.npy'
# 
# // Channel identities: 
# // 1. rf amplifier output
# // 2. hydrophone probe
# // 3. current monitor for e1
# // 4. current monitor for e2
# // 5. v mon e1 x10  
# // 6. v mon e2 x10 
# 
t_offset = 0.1
# // 7. tiepie voltage output waveform x10 
# // 8. diff input voltage across 1k resister. x 10
Fs = 5e6
duration = 3.0
N = Fs*duration
print("expected no. samples:",N)
t = np.linspace(0, duration, int(N), endpoint=False)
resistor_current_mon = 49.9  #  49.9 Ohms for current monitor, 1k resistor 

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
lowcut  = 5.0 # 
highcut = 1000 #  
sos_diff = signal.iirfilter(17, [lowcut, highcut], rs=60, btype='band',
                       analog=False, ftype='cheby2', fs=Fs,
                       output='sos')
# w, h = signal.sosfreqz(sos, 2000, fs=Fs)

lowcut  = 999990 # 
highcut = 1000000 #  
sos_sum = signal.iirfilter(17, [lowcut, highcut], rs=60, btype='band',
                       analog=False, ftype='cheby2', fs=Fs,
                       output='sos')


ae_idx = 10
e_idx  = 499990

timestep = 1.0/Fs
N = int(N)
# w = blackman(int(N)) # use the entire data. 
xf = np.fft.fftfreq(N, d=timestep)[:N//2]
frequencies = xf[1:N//2]


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx],idx

element,ae_idx = find_nearest(frequencies,ae_idx)
# print ('ae idx is: ',element,ae_idx)
element,e_idx = find_nearest(frequencies,e_idx)
# print ('e idx is: ',element,e_idx)

element,t_idx = find_nearest(t,t_offset)
print ('t idx: ', t_idx)

rf_channel  = 0    
hydrophone_channel = 1 
i_channel   = 3
v_channel   = 4 
ae_channel  = 6  
e_channel   = 7 

# e_channel = 5

resistor_current_mon = 49.9 

gain           = 100



d = np.load(filename)
a,b,c = d.shape
data = d.transpose(1,0,2).reshape(b,-1) 
print (data.shape)

t = t[t_idx:]
data = data[:,t_idx:]
chans, N = data.shape 
print ('new data shape',data.shape)
xf = np.fft.fftfreq(N, d=timestep)[:N//2]
frequencies = xf[1:N//2]


element,ae_idx = find_nearest(frequencies,10)
print ('ae idx is: ',element,ae_idx)
element,e_idx = find_nearest(frequencies,499990)
print ('e idx is: ',element,e_idx)
print ('ae_idx,e_idx',element,ae_idx,element,e_idx)

fft_aefield = fft(data[ae_channel])
fft_ae = np.abs(2.0/N * (fft_aefield))[1:N//2]
ae_V_pp_fft = fft_ae[ae_idx]*2
# AE_V_x = ae_V_pp_fft 
# AE_E_x = (AE_V_x/AE_gap)/gain
# print ('AE V_x (V) AE E_x (V/m): ',AE_V_x,AE_E_x)
ae_filtered = signal.sosfilt(sos_diff, data[ae_channel]) 

ae_sum_filtered = signal.sosfilt(sos_sum, data[ae_channel]) 

artifact = 2000000
ae_filter_artifact_removed = ae_filtered[artifact:]
ae_voltage_pp = np.max(ae_filter_artifact_removed) - np.min(ae_filter_artifact_removed)
# ae_voltage_pp = np.pi*sum(np.fabs(ae_filtered))/len(ae_filtered)
# AE_scale = (ae_voltage_pp/AE_gap)/gain
# print ('AE V_x filtered pp (V)',ae_voltage_pp )

fft_efield = fft(10*data[e_channel])
fft_e = np.abs(2.0/N * (fft_efield))[1:N//2]
V_pp_fft = fft_e[e_idx]*2
# E_x = V_pp_fft/AE_gap
# print ('V_x(V) E_x(V/m): ',V_pp_fft,E_x)

# print ('ratio: E to AE:',E_x/AE_E_x)


# print ('------------PROBE V,I,R----------------')
# V     = 10*data[v_channel]
# I     = 5*data[i_channel]/resistor_current_mon
# V_pp  = np.pi*sum(np.fabs(V))/len(V)
# I_pp  = np.pi*sum(np.fabs(I))/len(I)
# impedance = np.divide(V,I, where=I!=0)
# impedance = np.abs(np.median(impedance))
# print ('V_pp, I_pp(mA),R',V_pp, I_pp*1000,impedance)

fft_rf = fft(data[rf_channel])
fft_rf = np.abs(2.0/N * (fft_rf))[1:N//2]


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

# Change global font settings to help stylize the graph. 
plt.rc('font', family='serif')
plt.rc('font', serif='Arial')
plt.rcParams['axes.linewidth'] = 2




fig = plt.figure(figsize=(4,4))
ax = fig.add_subplot(111)
plt.plot(frequencies/1000, fft_rf, color = tableaucolorblind10[0],linewidth=2)
plt.plot(frequencies/1000, fft_e, color = tableaucolorblind10[5],linewidth=2)
plt.xlim([499.8,500.1])
plt.ylim(0,np.max(fft_rf))
plt.xticks(np.arange(499.8, 500.1, 0.1))
# plt.xticks([])
plt.yticks(fontsize=14)
plt.xticks(fontsize=14)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
save_title = 'df_fft_input_signals'
plt.savefig(save_title+".svg", format="svg") 
plt.savefig(save_title+".png")
plt.show()


fig = plt.figure(figsize=(3,4))
ax = fig.add_subplot(1,1,1) # multiply by 10000 to convert to mv. 
plt.plot(frequencies, 1000*fft_ae/gain, color = tableaucolorblind10[1],linewidth=2)
plt.xlim([0,40])
plt.ylim([0,0.35])
plt.xticks(fontsize=14)    
plt.yticks(fontsize=14) 
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
save_title = 'df_fft'
plt.tight_layout()
# fig.subplots_adjust(right=0.5, hspace=0.5)
# plt.savefig(save_title+".svg", format="svg") 
plt.savefig(save_title+".png", bbox_inches='tight')
plt.show()


# fig = plt.figure(figsize=(2,4))
# ax = fig.add_subplot(1,1,1)
# plt.plot((frequencies+10), 1000*fft_ae/gain, color = tableaucolorblind10[1],linewidth=2)
# plt.xlim([999980,1000000])
# plt.xticks([999990],['999990']) # 0.0002 is 0.2ms. 
# ax.yaxis.tick_right()
# plt.ylim([0,0.01])
# plt.xticks(fontsize=14)    
# plt.yticks(fontsize=14) 
# ax.spines['left'].set_visible(False)
# ax.spines['top'].set_visible(False)
# plt.tight_layout()
# save_title = 'sf_fft'
# # plt.savefig(save_title+".svg", format="svg") 
# plt.savefig(save_title+".png", bbox_inches='tight')
# plt.show()

df_chan = data[ae_channel]-np.mean(data[ae_channel])

fig, (ax2,ax3,ax4) = plt.subplots(nrows=3, sharex=True, subplot_kw=dict(frameon=False),figsize=(8,4)) # frameon=False removes frames
plt.subplots_adjust(hspace=.0)
# ax1.plot(t,ae_filtered, color = tableaucolorblind10[3])
ax2.plot(t,df_chan, color = tableaucolorblind10[1])
ax3.plot(t,data[e_channel], color = tableaucolorblind10[5])
ax4.plot(t,data[rf_channel], color = tableaucolorblind10[0])

ax2.set_xticks([])
ax3.set_xticks([])
ax2.tick_params(axis='y', labelsize=14)
ax3.tick_params(axis='y', labelsize=14)
ax4.tick_params(axis='y', labelsize=14)
ax2.set_ylim([-0.1,0.1])
plt.xlim([1.5,2.0])
plt.xticks([1.5,1.6,1.7,1.8,1.9,2.0],['0','0.1','0.2','0.3','0.4','0.5']) # 0.0002 is 0.2ms. 
plt.xticks(fontsize=14, rotation=0)
save_title = 'df_waveform_timeseries'
plt.savefig(save_title+".svg", format="svg") 
plt.savefig(save_title+".png")
plt.show()



