"""
Display

"""
import sys
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.tri as tri
import matplotlib.pyplot as plt
from scipy.io import loadmat
import matplotlib
import matplotlib.cm as cm
from scipy.stats.stats import pearsonr
from scipy import interpolate
from numpy.linalg import norm
from scipy.misc import derivative
from scipy.interpolate import interp1d
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.widgets import Slider
# from matplotlib.widgets import Textbox
from mpl_toolkits.axes_grid1 import AxesGrid
from tkinter import *
import matplotlib.colors as colors
from scipy.ndimage.filters import gaussian_filter1d
import cv2
from matplotlib.widgets import TextBox

# set the colormap and centre the colorbar
class MidpointNormalize(colors.Normalize):
    """
    Normalise the colorbar so that diverging bars work there way either side from a prescribed midpoint value)

    e.g. im=ax1.imshow(array, norm=MidpointNormalize(midpoint=0.,vmin=-100, vmax=100))
    """
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))


filename	="ae_xy_syncdata_0.npz"   # save out the data. 
data 		= np.load(filename) 	#  
ae0_diff_array 	= data['ae_diff_array'] 	   # 
ae0_sum_array   = data['ae_sum_array']
e0_array        = data['e_array']
x 			= data['x']
y 			= data['y']
ae0_array = ae0_diff_array + ae0_sum_array
a,b,c    = ae0_array.shape

e0 = np.mean(e0_array)
print ('e0:',e0)


filename    ="ae_xy_syncdata_90_zref.npz"   # save out the data. 
data        = np.load(filename)     #  
ae90_diff_array   = data['ae_diff_array']        # 
ae90_sum_array    = data['ae_sum_array']
e90_array         = data['e_array']
ae90_array = ae90_diff_array + ae90_sum_array
x = data['x']

e90 = np.mean(e90_array)
print ('e90:',e90)

Fs          = 5e6
duration    = 0.1
N = int(Fs*duration)
t  = np.linspace(0, duration, int(N), endpoint=False)
print ('ae_array shape:',ae0_array.shape)  # 30,31,50000
# Scroll through time of the image... 
rectangle           = [-10.0,10.0,-10.0,10.0]  #  full XY scan at zero degrees.  


idx = 95535 
colormap_choice = matplotlib.cm.bwr

mid_val  = 0.0
elev0_max = np.amax(ae0_array)
elev0_min = np.amin(ae0_array)

elev90_max = np.amax(ae90_array)
elev90_min = np.amin(ae90_array)
print ('elevvvv',elev0_min,elev0_max)
#############################
view_width = 100
# low pass filter the image. 
r = 5 #7 # convolution kernel size of the moving window. 
ham = np.hamming(b)[:,None] # 1D hamming
ham2 = np.hamming(a)[:,None] # 1D hamming
ham2d = np.sqrt(np.dot(ham2, ham.T)) ** r # expand to 2D hamming
# print ('ham2d',ham2d.shape)
#  
#  at 1MPa, i have a 1:500 ratio, with my current transducer. at 6V. 
#  at a higher pressure the ratio would be better.
#  
fig = plt.figure(figsize=(10,7),num ='Time scroller tool for XY data')

ax = fig.add_subplot(321)
fig.subplots_adjust(bottom=0.15) 
im_h = plt.imshow(ae0_array[:, :, idx],cmap=colormap_choice,interpolation='nearest',extent=rectangle,clim=(elev0_min, elev0_max), norm=MidpointNormalize(midpoint=mid_val,vmin=elev0_min, vmax=elev0_max))
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="10%", pad=0.1)
cbar = plt.colorbar(im_h, cax=cax)
ax.title.set_text('$\Phi_{AE}$(V) @ $\Delta$f')
# ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_ticks([])
ax.set_xlabel('Distance(mm)')
# ax.axis('off')

# timeseries plot of the wave form 
sum_array = np.sum(abs(ae0_array),axis=2)
m1,m2=np.where(sum_array==sum_array.max())
m1 = m1[0]
m2 = m2[0]
print ('max indice:', m1,m2)




ax2 = fig.add_subplot(322)
plt.plot(t,ae0_array[m1,m2, :])
if idx> view_width and idx < (N-view_width):
    ax2.set_xlim([t[idx-view_width],t[idx+view_width]   ])
elif (idx <=  view_width):
    ax2.set_xlim([0,t[idx+view_width]  ])
else:
    ax2.set_xlim([t[idx-view_width], t[N-1] ])
ax2.yaxis.tick_right()
ax2.yaxis.set_label_position("right")
asp = np.diff(ax2.get_xlim())[0] / np.diff(ax2.get_ylim())[0]
ax2.set_aspect(asp)
# ax2.axis('off')
# ax2.set_ylim([])

sum90_array = np.sum(abs(ae90_array),axis=2)
m190,m290=np.where(sum90_array==sum90_array.max())
m190 = m190[0]
m290 = m290[0]
print ('max indice:', m190,m290)

ax3 = fig.add_subplot(323)
fig.subplots_adjust(bottom=0.15) 
im_h2 = plt.imshow(ae90_array[:, :, idx],cmap=colormap_choice,interpolation='nearest',extent=rectangle,clim=(elev90_min, elev90_max), norm=MidpointNormalize(midpoint=mid_val,vmin=elev90_min, vmax=elev90_max))
divider = make_axes_locatable(ax3)
cax = divider.append_axes("right", size="10%", pad=0.1)
cbar = plt.colorbar(im_h2, cax=cax)
# ax.title.set_text('$\Phi_{AE}$(V) @ $\Delta$f')
# ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_ticks([])
ax.set_xlabel('Distance(mm)')

ax4 = fig.add_subplot(324)
plt.plot(t,ae90_array[m190,m290, :])
if idx> view_width and idx < (N-view_width):
    ax4.set_xlim([t[idx-view_width],t[idx+view_width]   ])
elif (idx <=  view_width):
    ax4.set_xlim([0,t[idx+view_width]  ])
else:
    ax4.set_xlim([t[idx-view_width], t[N-1] ])
ax4.yaxis.tick_right()
ax4.yaxis.set_label_position("right")
asp = np.diff(ax4.get_xlim())[0] / np.diff(ax4.get_ylim())[0]
ax4.set_aspect(asp)
# ax2.axis('off')
# ax2.set_ylim([])

line_idx = 21
ax5 = fig.add_subplot(325)
radial0_line   = ae0_array[:, line_idx, idx]
radial90_line  = ae90_array[:, line_idx, idx]
plt.plot(radial0_line,'b')
plt.plot(radial90_line,'r')

# plt.ax5(x=t[max_idx], linestyle=':', color='k')      

# update the figure with a change on the slider 
def update_depth(val):
    global idx
    idx = int(round(slider_depth.val))

    img = ae0_array[:, :, idx].T
    img_min = np.min(img)
    img_max = np.max(img)
    
    f = cv2.dft(img.astype(np.float32), flags=cv2.DFT_COMPLEX_OUTPUT)
    f_shifted = np.fft.fftshift(f)
    f_complex = f_shifted[:,:,0]*1j + f_shifted[:,:,1]
    f_filtered = ham2d * f_complex
    f_filtered_shifted = np.fft.fftshift(f_filtered)
    inv_img = np.fft.ifft2(f_filtered_shifted) # inverse F.T.
    filtered_img = np.real(inv_img)
    # print (np.min(filtered_img),np.max(filtered_img))
    norm_filtered = filtered_img/ (np.max(filtered_img)-np.min(filtered_img))    
    scaled_img0 = norm_filtered * (img_max-img_min)
    im_h.set_data(scaled_img0)
    # 
    # for updating the line graph 
    ax2.cla()
    ax2.plot(t, ae0_array[m1,m2, :])
    ax2.axvline(x=t[idx], linestyle=':', color='k') 
    fig.canvas.draw_idle()
    if idx> view_width and idx < (N-view_width):
        ax2.set_xlim([t[idx-view_width],t[idx+view_width]   ])
    elif (idx <=  view_width):
        ax2.set_xlim([0,t[idx+view_width]  ])
    else:
        ax2.set_xlim([t[idx-view_width], t[N-1] ])
    start,end = ax2.get_xlim()
    tick_values = [start,t[idx],end]
    ax2.xaxis.set_ticks(tick_values)
    ax2.set_xlabel('Time(s)')
    ax2.set_ylabel('$\Phi_{AE}$(V) @ $\Delta$f')


    img = ae90_array[:, :, idx].T
    img_min = np.min(img)
    img_max = np.max(img)
    f = cv2.dft(img.astype(np.float32), flags=cv2.DFT_COMPLEX_OUTPUT)
    f_shifted = np.fft.fftshift(f)
    f_complex = f_shifted[:,:,0]*1j + f_shifted[:,:,1]
    f_filtered = ham2d * f_complex
    f_filtered_shifted = np.fft.fftshift(f_filtered)
    inv_img = np.fft.ifft2(f_filtered_shifted) # inverse F.T.
    filtered_img = np.real(inv_img)
    # print (np.min(filtered_img),np.max(filtered_img))
    norm_filtered = filtered_img/ (np.max(filtered_img)-np.min(filtered_img))    
    scaled_img90 = norm_filtered * (img_max-img_min)
    # ax5.axvline(x=t[idx], linestyle=':', color='k') 
    im_h2.set_data(scaled_img90)

    ax4.cla()
    ax4.plot(t, ae90_array[m1,m2, :])
    ax4.axvline(x=t[idx], linestyle=':', color='k') 
    fig.canvas.draw_idle()
    if idx> view_width and idx < (N-view_width):
        ax4.set_xlim([t[idx-view_width],t[idx+view_width]   ])
    elif (idx <=  view_width):
        ax4.set_xlim([0,t[idx+view_width]  ])
    else:
        ax4.set_xlim([t[idx-view_width], t[N-1] ])
    start,end = ax4.get_xlim()
    tick_values = [start,t[idx],end]
    ax4.xaxis.set_ticks(tick_values)
    ax4.set_xlabel('Time(s)')
    ax4.set_ylabel('$\Phi_{AE}$(V) @ $\Delta$f')

    ax5.cla()
    # ax5.plot(t, ae90_array[m1,m2, :])
    # ax5.axvline(x=t[idx], linestyle=':', color='k') 
    radial0_line   = scaled_img0[:, line_idx]
    radial90_line  = scaled_img90[:, line_idx]
    ax5.plot(radial0_line,'b')
    ax5.plot(radial90_line,'r')


# setup a slider axis and the Slider
ax_depth = plt.axes([0.09, 0.02, 0.56, 0.04])
slider_depth = Slider(ax_depth, 'time idx: ', 0, N-1, valinit=round(idx),valfmt='%d' )
# ax_depth.axis('off')

def on_press(event):
    print('press', event.key)
    sys.stdout.flush()
    if event.key == 'k' or event.key=='left':
        print ('idx',idx)
        num = idx - 1
        print ('num',num) 
        slider_depth.set_val(int( num) )        
    if event.key == 'l' or event.key=='right':        
        print ('idx',idx)
        num = idx + 1
        print ('num',num) 
        slider_depth.set_val( int(num) )

fig.canvas.mpl_connect('key_press_event', on_press)

def submit(expression):
    slider_depth.set_val(int(expression) )

axbox = fig.add_axes([0.9, 0.02, 0.06, 0.04])
text_box = TextBox(axbox, "Go to idx: ")
text_box.on_submit(submit)
# text_box.set_val("0")  # Trigger `submit` with the initial string.

# for x in ax.ravel():
#     x.axis("off")
slider_depth.on_changed(update_depth)
plt.show()

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

# scroll through the data, and pick say 5 time indexes. 
# blue on left, blue in middle.
# 
index_array = [222325,222335,222345,255365,255375,255385,255395,255405,255415,255425,255435]
# Now, plot their mean and standard deviation. 
lines90 = []
lines0  = []
for i in range(len(index_array)):
    idx     = index_array[i]
    line0   = -(ae0_array[:,line_idx,idx])
    line0   = (line0 - np.min(line0))/(np.max(line0)-np.min(line0))
    line90  = (ae90_array[:,line_idx,idx])
    line90  = (line90 - np.min(line90))/(np.max(line90) - np.min(line90))
    lines0.append(line0)
    lines90.append(line90)
lines0  = np.array(lines0)
lines90 = np.array(lines90)
# 
meanline0   = np.mean(lines0,axis=0) 
meanline90  = np.mean(lines90,axis=0) 
stdline0    = np.std(lines0,axis=0)
stdline90   = np.std(lines90,axis=0)
# Have both lones start at 0. 
meanline0   = meanline0-meanline0[0]
meanline90  = meanline90-meanline90[0]

print ('x',x,len(x),len(meanline0))
x           = x[1:] # to account for the off by one thing. 
offset      = 1.5
offset2     = 1 
sigma_val   = 3
b           = 0.1
fig = plt.figure(figsize=(4,4))
ax  = fig.add_subplot(111)
plt.plot(x-offset,gaussian_filter1d(meanline0,sigma=sigma_val),color=tableaucolorblind10[3],linewidth=2)
plt.plot(x-offset2,gaussian_filter1d(meanline90,sigma=sigma_val),color=tableaucolorblind10[5],linewidth=2)
# now plot the standard deviation. 
plt.fill_between(x-offset,gaussian_filter1d(meanline0 - stdline0, sigma=sigma_val),
    gaussian_filter1d(meanline0 + stdline0 , sigma=sigma_val),color=tableaucolorblind10[3],alpha=0.5)
plt.fill_between(x-offset2,gaussian_filter1d(meanline90 - stdline90, sigma=sigma_val),
    gaussian_filter1d(meanline90 + stdline90, sigma=sigma_val),color=tableaucolorblind10[5],alpha=0.5)
plt.xlim([-10.0,10.0])
plt.xticks(fontsize=16)    
plt.xticks([]) 
plt.yticks([]) 
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_position('zero')
# ax.spines['bottom'].set_position('zero')
ax.spines['bottom'].set_position('data',-0.1)

legend_string = ['$0^o$','$90^o$']
plt.legend(legend_string,framealpha=0.0,frameon=False,loc="upper left",fontsize=14)
save_title = 'orientation' 
# plt.savefig(save_title+".svg", format="svg") 
plt.savefig(save_title+".png") 
plt.show()
# 
# Find the maximum index in each of the XY cross_sections. 
# Save these out as a time series, with they constituant e values. 
# For 0 element, 20,20, for 90 degree it is off center at: 22
# Check I can get a decent Pk value. i.e. did I get the electric field channel right?
# 