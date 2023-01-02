'''

Author: Jean Rintoul
Date: 29/12/2022

Description: Frequenxy mixing plot

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
filename = 'stream1.npy'
# // Channel identities: 
# // 1. rf amplifier output
# // 2. hydrophone probe
# // 3. current monitor for e1
# // 4. current monitor for e2
# // 5. v mon e1 x10  
# // 6. v mon e2 x10 
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
lowcut  = 492000  -3000 # 504000
highcut = 492000 + 3000 #  514000
sos = signal.iirfilter(17, [lowcut, highcut], rs=60, btype='band',
                       analog=False, ftype='cheby2', fs=Fs,
                       output='sos')
w, h = signal.sosfreqz(sos, 2000, fs=Fs)

e_lowcut  = 8000 - 1000
e_highcut = 8000 + 2000
sos_efield = signal.iirfilter(17, [e_lowcut, e_highcut], rs=60, btype='band',
                       analog=False, ftype='cheby2', fs=Fs,
                       output='sos')
w_e, h_e = signal.sosfreqz(sos, 2000, fs=Fs)                       


ae_idx = 500000-8000
e_idx  = 8000

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

rf_channel  = 0    
hydrophone_channel = 1 
i_channel   = 3
v_channel   = 4 
ae_channel  = 6  
e_channel   = 7 

resistor_current_mon = 49.9 
E_gap          = 0.0025
AE_gap         = 0.0014
gain           = 1000



d = np.load(filename)
a,b,c = d.shape
data = d.transpose(1,0,2).reshape(b,-1) 
# print (data.shape)

fft_aefield = fft(data[ae_channel])
fft_ae = np.abs(2.0/N * (fft_aefield))[1:N//2]
# ae_V_pp_fft = fft_ae[ae_idx]*2
# AE_V_x = ae_V_pp_fft 
# AE_E_x = (AE_V_x/AE_gap)/gain
# print ('AE V_x (V) AE E_x (V/m): ',AE_V_x,AE_E_x)
ae_filtered = signal.sosfilt(sos, data[ae_channel])
# ae_voltage_pp = np.pi*sum(np.fabs(ae_filtered))/len(ae_filtered)
# AE_scale = (ae_voltage_pp/AE_gap)/gain
# print ('AE V_x filtered (V) AE E_x filter (V/m)',ae_voltage_pp,(ae_voltage_pp/AE_gap)/gain )

AE = 2*fft_ae[ae_idx]/gain
print ('AE',AE)
# V_pp_fft = fft_e[e_idx]*2
# E_x = V_pp_fft/AE_gap
# print ('V_x(V) E_x(V/m): ',V_pp_fft,E_x)
e_filtered = signal.sosfilt(sos_efield, 10*data[7])

fft_efield = fft(10*e_filtered)
fft_efield = fft(10*data[e_channel])
fft_e = np.abs(2.0/N * (fft_efield))[1:N//2]
print ('E',2*fft_e[e_idx])
E = 2*fft_e[e_idx]

pK = ( AE/E ) / (1e6*1e-12)
print ('pK estimate', pK )

print ('------------PROBE V,I,R----------------')
V     = 10*data[v_channel]
I     = 5*data[i_channel]/resistor_current_mon
V_pp  = np.pi*sum(np.fabs(V))/len(V)
I_pp  = np.pi*sum(np.fabs(I))/len(I)
impedance = np.divide(V,I, where=I!=0)
impedance = np.abs(np.median(impedance))
print ('V_pp, I_pp(mA),R',V_pp, I_pp*1000,impedance)

fft_rf = fft(data[rf_channel])
fft_rf = np.abs(2.0/N * (fft_rf))[1:N//2]

fft_v = fft(data[v_channel])
fft_v = np.abs(2.0/N * (fft_v))[1:N//2]

fft_eappliedfield = fft(10*data[v_channel])
fft_eappliedfield = np.abs(2.0/N * (fft_eappliedfield))[1:N//2]

fft_ifield = fft(5*data[i_channel]/resistor_current_mon)
fft_ifield = np.abs(2.0/N * (fft_ifield))[1:N//2]
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
plt.xlim([1,1000])
plt.ylim(0,np.max(fft_rf))
stepsize = 500000
start, end = ax.get_xlim()
ax.set_xscale('log')
plt.yticks(fontsize=14)
plt.xticks(fontsize=14)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
save_title = 'fft_input_signals'
plt.savefig(save_title+".svg", format="svg") 
plt.savefig(save_title+".png")
plt.show()

fig = plt.figure(figsize=(4,4))
ax = fig.add_subplot(111)
plt.plot(frequencies/1000, fft_ae, color = tableaucolorblind10[1],linewidth=2)
plt.xlim([0,1100])
plt.ylim([0,0.12])
# stepsize = 500000
start, end = ax.get_xlim()
# ax.xaxis.set_ticks(np.arange(start, end, stepsize))
plt.ylim(0,np.max(fft_ae))
# ax.legend(['AE FFT(V)'],loc='upper left',framealpha= 0.0,fontsize=14)
# plt.ylabel('Volts(V)',fontsize=14)
# plt.xlabel('Frequency(Hz)',fontsize=14)
plt.yticks(fontsize=14)
plt.xticks(fontsize=14)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
save_title = 'fft_mixed_signal'
plt.savefig(save_title+".svg", format="svg") 
plt.savefig(save_title+".png")
plt.show()


fig = plt.figure(figsize=(4,4))
ax = fig.add_subplot(111)
plt.plot(frequencies/1000, fft_ae, color = tableaucolorblind10[1],linewidth=2)
# plt.ylim([0,0.045])
plt.xlim(485, 515)
plt.ylim(0,np.max(fft_ae))
plt.yticks(fontsize=14)
plt.xticks(fontsize=14)
# plt.xticks([])
# plt.yticks([])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
save_title = 'fft_mixed_signal_closeup'
# plt.savefig(save_title+".svg", format="svg") 
plt.savefig(save_title+".png")
plt.show()

# 
# 
# 
# 
# idx_rf_final = np.argmax(10*data[rf_channel] > 20)
# print ('idx_rf_final',idx_rf_final)
# lag =  idx_rf_final - idx_rf 
# t_offset = timestep * lag
# print ('t_offset',t_offset)
# it goes through the zero point, at the minimas. 
# tlag = 0.499880-0.499870
tlag = 0.000015
print ('tlag',tlag)
# 
fig, (ax1,ax2,ax3) = plt.subplots(nrows=3, sharex=True, subplot_kw=dict(frameon=False),figsize=(8,4)) # frameon=False removes frames
plt.subplots_adjust(hspace=.0)
# ax1.plot(t+t_offset,p_d,color = tableaucolorblind10[0]) # hydrophone
ax1.plot(t,data[ae_channel], color = tableaucolorblind10[1])
# ax2.plot(t+tlag,data[e_channel], color = tableaucolorblind10[5])
ax2.plot(t,data[e_channel], color = tableaucolorblind10[5])
ax3.plot(t,data[rf_channel], color = tableaucolorblind10[0] )
plt.xlim([0.4998,0.500])
# ax6.set_yticks([])
ax1.set_xticks([])
ax2.set_xticks([])
# ax3.set_xticks([])
# ax1.set_xlim([0.0,0.0002])
# ax2.set_xlim([0.0,0.0002])
# ax3.set_xlim([0.0,0.0002])
# 
# at measurement probe 0.2v pp
# ae fild is 0.5mV pp. 
# factor of 400. 
# 
plt.xticks([0.4998,0.4999,0.500],['0','0.1','0.2']) # 0.0002 is 0.2ms. 
plt.xticks(fontsize=14, rotation=0)
# ax1.set_yticklabels(fontsize=14)
# ax2.set_yticklabels(fontsize=14)
# ax3.set_yticklabels(fontsize=14)

ax1.tick_params(axis='y', labelsize=14)
ax2.tick_params(axis='y', labelsize=14)
ax3.tick_params(axis='y', labelsize=14)
# ax3.set_yticks(fontsize=14, rotation=0)
# ax2.set_yticks(fontsize=14, rotation=0)
# ax1.set_yticks(fontsize=14, rotation=0)
# ax1.set_yticks([])
# ax2.set_yticks([])
# ax3.set_yticks([])
# ax4.set_xticks([])
# plt.ylim([-0.3,0.35])
save_title = 'waveform_closeup'
# plt.savefig(save_title+".svg", format="svg") 
plt.savefig(save_title+".png")
plt.show()

