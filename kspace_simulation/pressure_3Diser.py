"""

Author: Jean Rintoul
Created: 25/04/2022. 
Last Updated: 
 	Updater: Jean Rintoul
 	Date: 31/12/2022

Description: Takes the output of the analytic Bessel spherical transducer solver, and turns it into a three dimensional field based on axial symmetry. 

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
# from scipy.fft import fft, fftfreq,fftn,fft2
from numpy.fft import fftn, fft2, fft, fftshift,ifftn,fftfreq
from scipy.integrate import quad
from scipy.io import loadmat
from scipy import interpolate
from pylab import*


frequency = 2.0
print ('frequency',str(frequency))
pressure_at_source = 1e6 # 1MPA We will use this to scale the gain factor. 
# I then extract the pressure field in complex phasor form. 
mdata 			= loadmat('large_area_'+str(frequency)+'Mhz.mat');
pp 				= mdata['pp'];
matp_x 			= mdata['xx'][0];
matp_z 			= mdata['z'][0];
pressure_amp 	= mdata['pressure_amp'];
mat_frequency 	= mdata['f0'];
pp 				= pressure_at_source*pp # scale the quantity from gain value into Pascals. 

print (pp.shape)
# find the focus. 
[fx,fz] = np.where(pressure_amp == np.amax(pressure_amp)) # 100, 103
print ('Focal pt of matlab script',fx[0],fz[0])
print ('pressure max(Pa)',pp[fx[0],fz[0]])
print ('pp',pp.shape,pp[fx[0],fz[0]])
# pp = 1e6*pp / pp[fx,fz] # Max pressure is now 1MPa. 
pp = 1e6*pp / pp[fx[0],fz[0]] # Max pressure is now 1MPa. 
print ('pressure max(Pa) normalized',pp[fx,fz])
# Thus, the 1D radial and axial components can be extracted, and plotted. 
print ('Focal Point in Complex Form: ',pp[fx,fz])
length_pp,axial_length = pp.shape
midpoint = int(length_pp/2)
print ('midpoint',midpoint)

def interpolator_complex(m):
    return interpolator_real(m) + 1j*interpolator_imag(m)

dx 		= 0.0001
xx, yy 	= np.meshgrid(matp_x,matp_x)
P_3D 	= np.zeros((length_pp,length_pp, axial_length),dtype=complex)
P_2D 	= np.zeros((length_pp,length_pp),dtype=complex) 

n= 0
for k in range(axial_length):
	n=n+1
	# print('iteration:',n)
	radial_p_phasor = pp[:,k] # this is the quantity to manipulate.
	# print ('len radial phasor',len(radial_p_phasor))
	x=matp_x[midpoint:]
	y=radial_p_phasor[midpoint:]
	interpolator_real = interpolate.interp1d(x, np.real(y))
	interpolator_imag = interpolate.interp1d(x, np.imag(y))

	for i in range(length_pp): 
	  for j in range(length_pp): 
	    # Find the distance of i and j from the center point. 
	    x_dist = (midpoint-j) 
	    y_dist = (midpoint-i) 
	    current_radius = np.sqrt(x_dist**2+y_dist**2)*dx
	    if current_radius < midpoint*dx:
	      P_2D[i, j] = interpolator_complex(current_radius)

	P_3D[:,:,k] = P_2D

print ('shape of P 2D: ',P_2D.shape)
print ('shape of P 3D: ',P_3D.shape)
# output the file the same as it was previously. 
finaldatafile = 'pressure'+str(frequency)+'MHz.npz'
np.savez(finaldatafile, P_3D = P_3D, matp_x=matp_x,matp_z=matp_z)

fig = plt.figure()
ax = fig.add_subplot(2,1,1)
plt.imshow(np.real(P_3D[200,:,:]),cmap='inferno')
ax.set_title('Pressure Magnitude ' )
ax2 = fig.add_subplot(2,1,2)
plt.imshow(np.real(P_3D[:,:,600]),cmap='inferno')

plt.show()


