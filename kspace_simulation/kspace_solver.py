"""

Author: Jean Rintoul
Date: 18.04.2022

Description: Solve the Poisson Equation, and Acoustoelectric result in k-space. 

This takes in a pressure file that has been converted into 3D space, to compute with an electric field at a given angle. 
Saves out all partial results into a large NPZ file for later analysis. 


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
from numpy.fft import fftn, fft2, fft, fftshift,ifftn,fftfreq
from scipy.integrate import quad
from scipy.io import loadmat
from scipy import interpolate
from pylab import*

images_folder = 'images/'

angle 		= 0   # 30,60,90 
frequency 	= 0.5


p_filename 	= 'pressure'+str(frequency)+'MHz.npz'
pdata 		= np.load(p_filename) # 
p_field_3d  = pdata['P_3D'][:,:,200:1000] # dont both with the extra z components. 
p_x 		= pdata['matp_x'] #394
p_y 		= pdata['matp_x'] #394
p_z 		= pdata['matp_z'][200:1000] #360

# chop it down
# p_field_3d = p_field_3d[:,:,200:1000]
print ('p field 3d size:',p_field_3d.shape)

# fig = plt.figure()
# ax = fig.add_subplot(2,1,1)
# plt.imshow(np.real(p_field_3d[50,:,:]),cmap='inferno')
# ax.set_title('Pressure Magnitude ' )
# plt.show()

x_len = max(p_x)-min(p_x)
y_len = max(p_y)-min(p_y)
z_len = max(p_z)-min(p_z)
dx 	  	   = 0.0001
incrementx = dx
incrementy = dx
incrementz = dx 
# xx, yy 	= np.meshgrid(matp_x,matp_x)
print ('increments:',incrementx,incrementy,incrementz)
print ('lengths:',x_len,y_len,z_len)


a,b,c = p_field_3d.shape
[presultx,presulty,presultz] = np.where(np.real(p_field_3d) == np.amax(np.real(p_field_3d) ) )
print ('max pressure is:',np.abs(p_field_3d[presultx,presulty,presultz]) )
print ('pressure max location in Pa is: ',presultx,presulty,presultz)

# https://en.wikipedia.org/wiki/Compressibility
# adiabatic compressibility of water. or Bulk Modulus: 1.96 x 109 Pa at https://moviecultists.com/is-the-bulk-modulus-of-water
# adiabatic_compressibility = 4.6e-10
# adiabatic_compressibility = 1e-9 # this is actually the k estimate from Xizi Songs acoustoelectric paper. 
# 1e-3 gave a k val of 5652813
# 1e-5 gave a k val of 565281
# 2e-8 gave a k val of 1131
# 
adiabatic_compressibility = 2e-8
# 
# So my mu term is 1.95e-8
# adiabatic compressibility is 4.6e-10
# 
# In this case, ionic mobility is 40 times larger than adiabatic compressibility. 
# 
# compute the gradient
grad_p = np.asarray(np.gradient(p_field_3d, p_x,p_y,p_z ))
# print ('grad p:',grad_p.shape)

grad_p = grad_p.reshape(3,-1,order="F")
# print ('first grad p',grad_p.shape)
grad_p = np.moveaxis(grad_p,0,-1)

# Now create the k vec. 
FreqCompX = fftfreq(len(p_x),d=incrementx)
FreqCompY = fftfreq(len(p_y),d=incrementy)
FreqCompZ = fftfreq(len(p_z),d=incrementz)
k_vec = np.zeros([len(p_x),len(p_y),len(p_z),3])
for i in range(len(p_x)):
	for j in range(len(p_y)):
		for k in range(len(p_z)):		
			k_vec[i,j,k,0] = FreqCompX[i]
			k_vec[i,j,k,1] = FreqCompY[j]
			k_vec[i,j,k,2] = FreqCompZ[k]
k_vec_shifted = fftshift(k_vec)
# print ('k_vec shifted:',k_vec_shifted)
k_vec_mag = np.linalg.norm(k_vec_shifted,axis=3)


# fig = plt.figure(figsize=(5,5))
# ax1 = fig.add_subplot(1,1,1)
# plt.imshow(k_vec_mag[:,:,presultz])
# plt.colorbar(im)
# plt.show()
# 

def polar2z(r,theta):
    return r * np.exp( 1j * theta )

def z2polar(z):
    return ( np.abs(z), np.angle(z) )

# Now create the E field phasor. 
f_carrier 		 = 500000.00
f_signal         = 8000.00
Fs       		 = 5e7
duration 		 = 1/f_carrier 
# print ('duration(s): ',duration)
N 				 = int(Fs *duration)
t 				 = np.linspace(0,duration,N)
# print ('number of points in t', len(t))
theta 	= 0 
E_size  = 3200
wc      = 2*np.pi*f_signal*t 
xm 		= E_size * np.exp(1j*(wc))

print ('xm:',xm[0],xm.shape)
E_max = np.real(xm[0]) # 
# Angular offset: this is if you want to manually create an e field. 
# E_max 		= 1000
degrees_90  = [E_max,0,0]
degrees_60 	= [E_max*np.sqrt(3)/2,0,E_max - E_max*np.sqrt(3)/2]
degrees_30 	= [E_max/2,0,E_max/2]
degrees_0 	= [0,0,E_max]
if angle == 0:
	angle_choice = degrees_0
elif angle == 30:
	angle_choice = degrees_30
elif angle == 60:
	angle_choice = degrees_60
else: 
	angle_choice = degrees_90

print ('angle choice:',angle_choice)
# Define a field in a single direction. 
E_0 = np.zeros([len(p_x),len(p_y),len(p_z),3])
for i in range(len(p_x)):
	for j in range(len(p_y)):
		for k in range(len(p_z)):
			E_0[i,j,k,0] = angle_choice[0] # 1000 # V/m 
			E_0[i,j,k,1] = angle_choice[1]
			E_0[i,j,k,2] = angle_choice[2]


em_inPgrid_flattened = E_0.reshape(len(p_x)*len(p_y)*len(p_z),3,order='F')

diffusion_source = - np.matmul(em_inPgrid_flattened[:,None,:], adiabatic_compressibility*grad_p[:,:,None]) [:,0]
# print ('completed matmul',diffusion_source.shape)
# Put E source in 3D so that we can visualize it. 
diffusion_source_3d = diffusion_source.reshape( len(p_x),len(p_y),len(p_z), order='F')

# phi_k = np.divide(fftn(diffusion_source_3d), k_vec_mag**2, out=phi_k, where=non_zero)
phi_k = np.divide(fftn(diffusion_source_3d), k_vec_mag*k_vec_mag, out=np.zeros_like(fftn(diffusion_source_3d)), where=(k_vec_mag*k_vec_mag)!=0)

# print ('phi(k)',phi_k.shape)
phi_space = ifftn(phi_k)
# print ('phi(space)',phi_space.shape)
grad_phi = np.asarray(np.gradient(phi_space,p_x,p_y,p_z))
grad_phi = np.moveaxis(grad_phi,0,-1)
# print ('grad phi:',grad_phi.shape)

AE 		= grad_phi
# output the file the same as it was previously. 
finaldatafile = 'check_ae_ephasor_kspace_a'+str(angle)+'degrees_f'+str(frequency)+'MHz.npz'
np.savez(finaldatafile, E = AE, phi = phi_space, x=p_x,y=p_y,z=p_z,sourceterm=diffusion_source_3d)

AE_mag 	= np.linalg.norm(AE,axis=3)
# print ('AE_mag:',AE_mag.shape,AE_mag[:,:,presultz])
AEsliceXY = AE_mag[:,:,presultz]
AEsliceXZ = AE_mag[:,presulty,:]
AEsliceXZ = np.moveaxis(AEsliceXZ,1,-1)
print (AEsliceXY.shape,AEsliceXZ.shape)
#
# print (AE.shape)   # 401, 401, 800, 3 
# a,b,c,d = AE.shape 
# print ('max = ',np.amax(AE[int((a-1)/2),int((b-1)/2),:]))
# print ('val at focus',AEsliceXY[int((a-1)/2),int((b-1)/2) ])
maxae = np.amax(np.real(AE))
# maxae = AEsliceXY[50,50][0] # np.amax(AEsliceXZ)
print ('max ae value:',maxae)
K = maxae/(1e6*E_max*1e-12)
print ('efficiency pico val:',round(K) )
# 
# plot to see what it came up with in the end. 
fig = plt.figure(figsize=(4,7))
ax1 = fig.add_subplot(2,1,1)
plt.imshow(AEsliceXY,cmap='inferno',interpolation='nearest')
ax1.title.set_text('AE XY slice')
plt.colorbar()
# ax1.axis("off")
ax2 = fig.add_subplot(2,1,2)
plt.imshow(AEsliceXZ,cmap='inferno',interpolation='nearest')
ax2.title.set_text('AE XZ slice')
# ax2.axis("off")
plt.colorbar()
plt.savefig(images_folder+"angle_"+str(angle)+"frequency"+str(frequency)+"fourier_poisson.png") 
plt.show()


