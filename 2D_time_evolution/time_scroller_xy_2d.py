"""
Author: Jean Rintoul 
Date: 02.01.2022
Description: GUI to scroll through time for the XY view of experimental acoustoelectric data. 

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
ae_diff_array 	= data['ae_diff_array'] 	   # 
# ae_sum_array    = data['ae_sum_array']
# ae_broadband_array = data['ae_broadband_array']
x 			= data['x']
y 			= data['y']

ae_array = ae_diff_array 
a,b,c = ae_array.shape

Fs          = 5e6
duration    = 0.1
N = int(Fs*duration)
t  = np.linspace(0, duration, int(N), endpoint=False)
print ('ae_array shape:',ae_array.shape)  # 30,31,50000
# Scroll through time of the image... 
rectangle           = [-10.0,10.0,-10.0,10.0]  #  full XY scan at zero degrees.  


idx = 95535 
colormap_choice = matplotlib.cm.bwr
elev_max = 0.11
elev_min = -0.11
mid_val  = 0.0
elev_max = np.amax(ae_array)
elev_min = np.amin(ae_array)
print ('elevvvv',elev_min,elev_max)
#############################
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
ax = fig.add_subplot(121)
fig.subplots_adjust(bottom=0.15) 

im_h = plt.imshow(ae_array[:, :, idx],cmap=colormap_choice,interpolation='nearest',extent=rectangle,clim=(elev_min, elev_max), norm=MidpointNormalize(midpoint=mid_val,vmin=elev_min, vmax=elev_max))
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="10%", pad=0.1)
cbar = plt.colorbar(im_h, cax=cax)
ax.title.set_text('$\Phi_{AE}$(V) @ $\Delta$f')
# ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_ticks([])
ax.set_xlabel('Distance(mm)')
# ax.axis('off')
# 
# TODO: which index has the maximum values? 
# timeseries plot of the wave form 
sum_array = np.sum(abs(ae_array),axis=2)
m1,m2=np.where(sum_array==sum_array.max())
m1 = m1[0]
m2 = m2[0]
print ('max indice:', m1,m2)
# maxidx = np.argmax(sum_array)
view_width = 100
ax2 = fig.add_subplot(122)
plt.plot(t,ae_array[m1,m2, :])
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

# update the figure with a change on the slider 
def update_depth(val):
    global idx
    idx = int(round(slider_depth.val))

    img = ae_array[:, :, idx]
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
    scaled_img = norm_filtered * (img_max-img_min)
    im_h.set_data(scaled_img)
    # 
    # for updating the line graph 
    ax2.cla()
    ax2.plot(t, ae_array[m1,m2, :])
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









