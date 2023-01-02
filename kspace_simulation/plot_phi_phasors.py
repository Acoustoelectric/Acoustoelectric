"""
Author: Jean Rintoul
Created: 25/04/2022. 
Last Updated: 
    Updater: Jean Rintoul
    Date: 31/12/2022

Description: Display the phasor components to observe the spatial distribution with respect to time and save an image file. 

"""
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
from findiff import Gradient, Divergence, Laplacian, Curl
from scipy.interpolate import interp1d



angle = 0 
filename="kae_ephasor_kspace_a0degrees_f0.5MHz.npz"
# filename="ae_kspace_a"+str(angle)+"degrees_f0.5MHz.npz"

save_title = "mix"+str(angle)+"phasorsXY"
print ('save title: ',save_title)

filename = filename
data = np.load(filename)
#print (data.keys())
E 	= data['E']
phi = data['phi']
x 	= data['x']
y 	= data['y']
z 	= data['z']
# print(E.shape,phi.shape)

# Trim off all the edges 
edgewidth = 0
E = E[edgewidth:(len(x)-edgewidth),edgewidth:(len(y)-edgewidth),edgewidth:(len(z)-edgewidth),:]
phi = phi[edgewidth:(len(x)-edgewidth),edgewidth:(len(y)-edgewidth),edgewidth:(len(z)-edgewidth)]
x = x[edgewidth:(len(x)-edgewidth)]
y = y[edgewidth:(len(y)-edgewidth)]
z = z[edgewidth:(len(z)-edgewidth)]

E_mag = np.linalg.norm(E,axis = 3) # get the magnitude.
[resultx,resulty,resultz] = np.where(np.real(E_mag) == np.amax(np.real(E_mag) ) )
# print ('max slice of laplace_f is: ',resultx,resulty,resultz)
z_slice  = resultz[0]
y_slice  = resulty[0]
x_slice  = resultx[0]
print (E_mag.shape)
print ('x slice:',x_slice)
xm, ym, zm = np.meshgrid(x,y,z)

def polar2z(r,theta):
    return r * np.exp( 1j * theta )

def z2polar(z):
    return ( np.abs(z), np.angle(z) )

'''

	phi processing for phasor frequency mixing phasor images
	- phi is a phasor, but was calculated with an assumed DC electric field. 


'''
import matplotlib.colors as colors 

# set the colormap at the centre the colorbar. 
class MidpointNormalize(colors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    # def __call__(self, value, clip=None):
    #     x, y = [self.vmin, self.midpoint, self.vmax], [0,0.5,1]
    #     return np.ma.masked_array(np.interp(value, x, y), np.isnan(value) )
    def __call__(self, value, clip=None):
        normalized_min = max(0, 1 / 2 * (1 - abs((self.midpoint - self.vmin) / (self.midpoint - self.vmax))))
        normalized_max = min(1, 1 / 2 * (1 + abs((self.vmax - self.midpoint) / (self.midpoint - self.vmin))))
        normalized_mid = 0.5
        x, y = [self.vmin, self.midpoint, self.vmax], [normalized_min, normalized_mid, normalized_max]
        return np.ma.masked_array(np.interp(value, x, y))


f_carrier 		 = 500000.00
f_signal         = 8000.00
Fs       		 = 5e7
duration 		 = 1/f_carrier 
# duration 		 = 1/f_signal
print ('duration(s): ',duration)
N 				 = int(Fs *duration)
t 				 = np.linspace(0,duration,N)
print ('number of points in t', len(t))
phi_slice = phi[:,:,x_slice]
# problem is this z2 polar takes the absolute value... which removes the sign info. 
rS, thetaS = z2polar(phi_slice)
zS = polar2z( rS, thetaS )

a,b = phi_slice.shape
phi_time = np.empty((a, b, len(t)),dtype = 'complex_')

for i in range(a): 
    for j in range(b): 
        wm      = 2*np.pi*f_carrier*t + thetaS[i,j]
        wc      = 2*np.pi*f_signal*t + thetaS[i,j]

        # Euler.
        # xm        = ((phi_slice[i,j])/2)*(np.exp(1j*(wm+wc)) +np.exp(1j*(wm-wc)) )
        # xm_diff   =  (phi_slice[i,j]/2)*(np.exp(1j*(wm-wc)) )
        # xm_sum    = (phi_slice[i,j]/2)*(np.exp(1j*(wm+wc)) )   

        # using cos product to sum trig identity.
        xm_diff =  (phi_slice[i,j]/2)*(np.cos(wm-wc) )
        xm_sum  =  (phi_slice[i,j]/2)*(np.cos(wm+wc) )           
        xm = xm_diff + xm_sum

        phi_time[i,j,:] = xm  #xm # xm_diff + xm_sum 

# print ('phi time: ',phi_time.shape,phi_time[100,100,10])
a,b,c = phi_time.shape
pt1  = 0
pt2  = int(c/8)-1
pt3  = int(2*c/8)-1
pt4  = int(3*c/8)-1
pt5  = int(4*c/8)-1
pt6  = int(5*c/8)-1
pt7  = int(6*c/8)-1
pt8  = int(7*c/8)-1
pt9  = int(c)-1

print ('phi_time min and max: ',np.min(np.real(phi_time)),np.max(np.real(phi_time))   )
# print ('phi_slice min and max: ',np.min(np.real(phi_slice)),np.max(np.real(phi_slice))   )
# Let's scale phi time to be the same magnitude as phi slice. 
# 
# fig = plt.figure(figsize=(5,5))
# ax1 = fig.add_subplot(111)
# elev_min = np.min(np.real(phi_slice))/2
# elev_max = np.max(np.real(phi_slice))/2
# mid_val = 0.0
# cs  = ax1.imshow(np.real(phi_slice),cmap='bwr',extent=[-7.5,7.5,-7.5,7.5],alpha=0.8,clim=(elev_min,elev_max), norm=MidpointNormalize(midpoint=mid_val,vmin=elev_min,vmax=elev_max) )
# cbar = fig.colorbar(cs,fraction=0.046, pad=0.04)
# # ax1.axis("off")
# plt.show()


# mappable = plt.cm.ScalarMappable(cmap=plt.cm.inferno)
# mappable.set_array(np.real(phi_time[:,:,5]))

# cmin = 0 
# cmax = np.max(np.real(phi_time) )

elev_min    = np.min(np.real(phi_time ))
elev_max    = np.max(np.real(phi_time))
mid_val     = 0.0
# rectangle   = [-10.0,10.0,-10.0,10.0]
# rectangle   = [-15.0,15.0,-15.0,15.0]
rectangle   = [-20.0,20.0,-20.0,20.0]
# print ('phi time min and max: ',np.min(np.real(phi_time )) ,np.max( np.real(phi_time) ))
xlim_min = -10.0
xlim_max = 10.0
ylim_min = -10.0
ylim_max = 10.0

fig = plt.figure(figsize=(12,4))
ax1 = fig.add_subplot(1,5,1)
cs = ax1.imshow(np.real(phi_time[:,:,pt1].T), cmap='bwr',extent=rectangle,alpha=0.8,clim=(elev_min,elev_max), norm=MidpointNormalize(midpoint=mid_val,vmin=elev_min,vmax=elev_max)  )
# cbar = fig.colorbar(cs,fraction=0.046, pad=0.04)
# ax1.set_title('$\lambda = 0 \pi$ ')
ax1.axis("off")
ax1.set_xlim([xlim_min,xlim_max])
ax1.set_ylim([ylim_min,ylim_max])

ax2 = fig.add_subplot(1,5,2)
cs = ax2.imshow(np.real(phi_time[:,:,pt3].T), cmap='bwr',extent=rectangle,alpha=0.8,clim=(elev_min,elev_max), norm=MidpointNormalize(midpoint=mid_val,vmin=elev_min,vmax=elev_max) )
# cbar = fig.colorbar(cs,fraction=0.046)
# ax2.set_title('$\lambda = \dfrac{\pi}{2}$ ')
ax2.axis("off")
ax2.set_xlim([xlim_min,xlim_max])
ax2.set_ylim([ylim_min,ylim_max])

ax3 = fig.add_subplot(1,5,3)
cs = ax3.imshow(np.real(phi_time[:,:,pt5].T), cmap='bwr',extent=rectangle,alpha=0.8,clim=(elev_min,elev_max), norm=MidpointNormalize(midpoint=mid_val,vmin=elev_min,vmax=elev_max)  )
# cbar = fig.colorbar(cs,fraction=0.046, pad=0.04)
# ax3.set_title('$\lambda = \pi$ ')
ax3.axis("off")
ax3.set_xlim([xlim_min,xlim_max])
ax3.set_ylim([ylim_min,ylim_max])

ax4 = fig.add_subplot(1,5,4)
cs = ax4.imshow(np.real(phi_time[:,:,pt7].T), cmap='bwr',extent=rectangle,alpha=0.8,clim=(elev_min,elev_max), norm=MidpointNormalize(midpoint=mid_val,vmin=elev_min,vmax=elev_max) )
# cbar = fig.colorbar(cs,fraction=0.046, pad=0.04)
# ax4.set_title('$\lambda = \dfrac{3 \pi}{2} $ ')
ax4.axis("off")
ax4.set_xlim([xlim_min,xlim_max])
ax4.set_ylim([ylim_min,ylim_max])

ax5 = fig.add_subplot(1,5,5)
cs = ax5.imshow(np.real(phi_time[:,:,pt9].T), cmap='bwr',extent=rectangle,alpha=0.8,clim=(elev_min,elev_max), norm=MidpointNormalize(midpoint=mid_val,vmin=elev_min,vmax=elev_max) )
# ax5.set_title('$\lambda = 2\pi$')
ax5.axis("off")
ax5.set_xlim([xlim_min,xlim_max])
ax5.set_ylim([ylim_min,ylim_max])



save_title = 'phi_wrt_time_a'+str(angle)+'_ef_'+str(f_signal) 
# plt.savefig(save_title+".svg", format="svg") 
# plt.savefig(save_title+".png")

# plt.savefig(save_title+".svg", format="svg",transparent=True,
#            pad_inches=0,bbox_inches='tight') 
plt.savefig(save_title+".png",transparent=True,
           pad_inches=0,bbox_inches='tight') 


plt.show()
