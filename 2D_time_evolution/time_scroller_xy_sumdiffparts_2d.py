"""
Author: Jean Rintoul 
Date: 02.01.2022
Description: Display the XY view w.r.t. time of the phantom acoustoelectric dataset, showing the difference, sum and superposition waves independently. 

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


filename	="ae_xy_syncdata_90_zref.npz"   # save out the data. 
# filename    ="ae_xy_syncdata_90.npz"      # save out the data. 
# filename    = "ae_xy_syncdata_90_offset.npz"

data 		= np.load(filename) 	#  
ae_diff_array 	= data['ae_diff_array'] 	   # 
ae_sum_array    = data['ae_sum_array']
ae_broadband_array = data['ae_broadband_array']
x 			= data['x']
y 			= data['y']
print ('x',x)
print ('size x and y:',len(x),len(y))
ae_array = ae_diff_array + ae_sum_array
# ae_array = ae_broadband_array
a,b,c = ae_array.shape

Fs          = 5e6
duration    = 0.1
N = int(Fs*duration)
t  = np.linspace(0, duration, int(N), endpoint=False)
print ('ae_array shape:',ae_array.shape)  # 30,31,50000
# Scroll through time of the image... 
rectangle           = [-10.0,10.0,-10.0,10.0]  #  full XY scan at zero degrees.  
# rectangle         = [-15.0,15.0,-15.0,15.0]  #  full XY scan at zero degrees. 

idx = 95535 
colormap_choice = matplotlib.cm.bwr
# elev_max = 0.004
# elev_min = -0.004
mid_val  = 0.0

factor = 0.4

sum_elev_max = factor*np.amax(ae_sum_array)
sum_elev_min = factor*np.amin(ae_sum_array)
print ('sum range:',sum_elev_min,sum_elev_max)

diff_elev_max = factor*np.amax(ae_diff_array)
diff_elev_min = factor*np.amin(ae_diff_array)
print ('diff range:',diff_elev_min,diff_elev_max)

ae_elev_max = factor*np.amax(ae_array)
ae_elev_min = factor*np.amin(ae_array)
print ('ae range:',ae_elev_min,ae_elev_max)
#  
# view width for the time series plot. 
view_width = 1000
view_width = 50 
#############################
# low pass filter the image. 
r = 5 #7 # convolution kernel size of the moving window. 
ham = np.hamming(b)[:,None] # 1D hamming
ham2 = np.hamming(a)[:,None] # 1D hamming
ham2d = np.sqrt(np.dot(ham2, ham.T)) ** r # expand to 2D hamming
# print ('ham2d',ham2d.shape)
#  
# at 1MPa, i have a 1:500 ratio, with my current transducer. at 6V. 
# at a higher pressure the ratio would be better.
#  
fig = plt.figure(figsize=(8,8),num ='Time scroller tool for XY data')
fig.subplots_adjust(bottom=0.15) 

ax = fig.add_subplot(321)
im_diff = plt.imshow(ae_diff_array[:, :, idx],cmap=colormap_choice,interpolation='nearest',extent=rectangle,clim=(diff_elev_min, diff_elev_max), norm=MidpointNormalize(midpoint=mid_val,vmin=diff_elev_min, vmax=diff_elev_max))
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="10%", pad=0.1)
cbar = plt.colorbar(im_diff, cax=cax)
ax.title.set_text('$\Phi_{AE}$(V) @ $\Delta$f')
ax.get_yaxis().set_ticks([])
# ax.set_xlabel('Distance(mm)')

# timeseries plot of the wave form 
sum_array = np.sum(abs(ae_diff_array),axis=2)
m1,m2=np.where(sum_array==sum_array.max())
m1 = m1[0]
m2 = m2[0]
print ('max indice:', m1,m2)
ax2 = fig.add_subplot(322)
plt.plot(t,ae_diff_array[m1,m2, :])
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



# Now do sum parts. 
ax3 = fig.add_subplot(323)
im_sum = plt.imshow(ae_sum_array[:, :, idx],cmap=colormap_choice,interpolation='nearest',extent=rectangle,clim=(sum_elev_min, sum_elev_max), norm=MidpointNormalize(midpoint=mid_val,vmin=sum_elev_min, vmax=sum_elev_max))
divider = make_axes_locatable(ax3)
cax = divider.append_axes("right", size="10%", pad=0.1)
cbar = plt.colorbar(im_sum, cax=cax)
ax3.title.set_text('$\Phi_{AE}$(V) @ $\Sigma$f')
ax3.get_yaxis().set_ticks([])
# ax3.set_xlabel('Distance(mm)')

# timeseries plot of the wave form 
sum_array = np.sum(abs(ae_sum_array),axis=2)
m1,m2=np.where(sum_array==sum_array.max())
m1 = m1[0]
m2 = m2[0]
print ('max indice:', m1,m2)
ax4 = fig.add_subplot(324)
plt.plot(t,ae_sum_array[m1,m2, :])
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


# Now do diff + sum parts. 
ax5 = fig.add_subplot(325)
im_ae = plt.imshow(ae_array[:, :, idx],cmap=colormap_choice,interpolation='nearest',extent=rectangle,clim=(ae_elev_min, ae_elev_max), norm=MidpointNormalize(midpoint=mid_val,vmin=ae_elev_min, vmax=ae_elev_max))
divider = make_axes_locatable(ax5)
cax = divider.append_axes("right", size="10%", pad=0.1)
cbar = plt.colorbar(im_ae, cax=cax)
ax5.title.set_text('$\Phi_{AE}$(V) ')
ax5.get_yaxis().set_ticks([])
ax5.set_xlabel('Distance(mm)')
print ('ae total min and max:',np.min(ae_array[:,:,idx]),np.max(ae_array[:,:,idx]))

# timeseries plot of the wave form 
sum_array = np.sum(abs(ae_array),axis=2)
m1,m2=np.where(sum_array==sum_array.max())
m1 = m1[0]
m2 = m2[0]
print ('max indice:', m1,m2)
ax6 = fig.add_subplot(326)
plt.plot(t,ae_array[m1,m2, :])
if idx> view_width and idx < (N-view_width):
    ax6.set_xlim([t[idx-view_width],t[idx+view_width]   ])
elif (idx <=  view_width):
    ax6.set_xlim([0,t[idx+view_width]  ])
else:
    ax6.set_xlim([t[idx-view_width], t[N-1] ])
ax6.yaxis.tick_right()
ax6.yaxis.set_label_position("right")
asp = np.diff(ax6.get_xlim())[0] / np.diff(ax6.get_ylim())[0]
ax6.set_aspect(asp)



# update the figure with a change on the slider 
def update_depth(val):
    global idx
    idx = int(round(slider_depth.val))

    img = ae_diff_array[:, :, idx].T
    img_min = np.min(img)
    img_max = np.max(img)
    f = cv2.dft(img.astype(np.float32), flags=cv2.DFT_COMPLEX_OUTPUT)
    f_shifted = np.fft.fftshift(f)
    f_complex = f_shifted[:,:,0]*1j + f_shifted[:,:,1]
    f_filtered = ham2d * f_complex
    f_filtered_shifted = np.fft.fftshift(f_filtered)
    inv_img = np.fft.ifft2(f_filtered_shifted) # inverse F.T.
    filtered_img = np.real(inv_img)
    norm_filtered = filtered_img/ (np.max(filtered_img)-np.min(filtered_img))    
    scaled_img = norm_filtered * (img_max-img_min)
    # print (np.min(filtered_img),np.max(filtered_img))
    im_diff.set_data(scaled_img)
    # 
    # for updating the line graph 
    ax2.cla()
    ax2.plot(t, ae_diff_array[m1,m2, :])
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
    # ax2.set_xlabel('Time(s)')
    ax2.set_ylabel('$\Phi_{AE}$(V) @ $\Delta$f')


    # Now update for the sum array
    img = ae_sum_array[:, :, idx].T
    img_min = np.min(img)
    img_max = np.max(img)
    f = cv2.dft(img.astype(np.float32), flags=cv2.DFT_COMPLEX_OUTPUT)
    f_shifted = np.fft.fftshift(f)
    f_complex = f_shifted[:,:,0]*1j + f_shifted[:,:,1]
    f_filtered = ham2d * f_complex
    f_filtered_shifted = np.fft.fftshift(f_filtered)
    inv_img = np.fft.ifft2(f_filtered_shifted) # inverse F.T.
    filtered_img = np.real(inv_img)
    norm_filtered = filtered_img/ (np.max(filtered_img)-np.min(filtered_img))    
    scaled_img = norm_filtered * (img_max-img_min)
    # print (np.min(filtered_img),np.max(filtered_img))
    im_sum.set_data(scaled_img)


    # for updating the line graph 
    ax4.cla()
    ax4.plot(t, ae_sum_array[m1,m2, :])
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
    # ax2.set_xlabel('Time(s)')
    ax4.set_ylabel('$\Phi_{AE}$(V) @ $\Sigma$f')

    # Now update for the ae array
    # ax5.cla()
    img = ae_array[:, :, idx].T
    # find the max pointin the array 
    p1,p2=np.where(img==img.max())
    print ('p1,p2:',p1[0],p2[0])
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
    im_ae.set_data(scaled_img)

    # print ('point is:',x[p1[0]],y[p2[0]])
    # f1,f2=np.where(filtered_img==filtered_img.max())
    # print ('f1,f2:',f1[0],f2[0])

    # spot = np.zeros((40,40))
    # # Make a region in y that we're interested in...
    # spot[4:6, 1:8] = 1
    # spot = np.ma.masked_where(spot == 0, spot)
    # # colormap_choice = matplotlib.cm.bwr
    # # plt.imshow(x, cmap=mpl.cm.bone)
    # im_ae.set_data(spot)

    # masked_data = np.empty((40,40))*np.nan
    # # max point in unfiltered data. 
    # masked_data[20,20] = 0.1
    # masked_data[19,20] = -0.1
    # masked_data[x[p1[0]],y[p2[0]]] = 0.3
    # # max point in filtered data. 
    # masked_data[x[f1[0]],y[f2[0]]] = -0.3
    # im_ae.set_data(masked_data)


    # put a red dot, size 40, at 2 locations:
    # plt.scatter(x=[x[p1[0]], 40], y=[50, 60], c='r', s=40)
    # ax5.plot(x[p1[0]],y[p2[0]], "og", markersize=5)  # og:shorthand for green circle
    # ax5.plot(x[f1[0]],y[f2[0]], "1r", markersize=5)  # og:shorthand for green circle
    # ax5.set_xlim([-10,10])
    # ax5.set_ylim([-10,10])


    # for updating the line graph 
    ax6.cla()
    ax6.plot(t, ae_array[m1,m2, :])
    ax6.axvline(x=t[idx], linestyle=':', color='k') 
    fig.canvas.draw_idle()
    if idx> view_width and idx < (N-view_width):
        ax6.set_xlim([t[idx-view_width],t[idx+view_width]   ])
    elif (idx <=  view_width):
        ax6.set_xlim([0,t[idx+view_width]  ])
    else:
        ax6.set_xlim([t[idx-view_width], t[N-1] ])
    start,end = ax6.get_xlim()
    tick_values = [start,t[idx],end]
    ax6.xaxis.set_ticks(tick_values)
    ax6.set_xlabel('Time(s)')
    ax6.set_ylabel('$\Phi_{AE}$(V)')
    # print to screen the min and max values: 
    print ('ae total min and max:',np.amin(ae_array[:,:,idx]),np.amax(ae_array[:,:,idx]))
    print ('ae filtered min and max:',np.amin(filtered_img),np.amax(filtered_img))   


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









