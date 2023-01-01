"""

Author: Jean Rintoul
Date: 28/12/2022
Purpose@ Display sub-terms from the acoustoelectric k-space simulation 

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
# from findiff import Gradient, Divergence, Laplacian, Curl
from scipy.interpolate import interp1d
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.widgets import Slider

angle 		= 0   # 30,60,90 
frequency 	= 0.5

save_title = str(angle)+"degrees_f"+str(frequency)
# print ('save_title:',save_title)
p_filename = 'pressure'+str(frequency)+'MHz.npz'
pdata = np.load(p_filename) 			#  
p_field_3d  = pdata['P_3D'] 			# 
p_x = pdata['matp_x'] 					# 
p_y = pdata['matp_x'] 					# 
p_z = pdata['matp_z'][200:1000] 		# 
p_field_3d = p_field_3d[:,:,200:1000] 	# 
[presultx,presulty,presultz] = np.where(np.real(p_field_3d) == np.amax(np.real(p_field_3d) ) )
grad_p = np.asarray(np.gradient(p_field_3d, p_x,p_y,p_z ))
grad_p = np.moveaxis(grad_p,0,-1)


print('p_z shape:',p_z.shape)
# print ('p_z',p_z*1000)
p_z_scale = p_z*1000
p_x_scale = p_x*1000
print ('p_z',np.min(p_z_scale),np.max(p_z_scale))
print ('p_x',np.min(p_x_scale),np.max(p_x_scale))


filename="ae_data_kspace_a"+str(angle)+"degrees_f"+str(frequency)+"MHz.npz"
data = np.load(filename)
E 	= data['E']
phi = data['phi']
x 	= data['x']
y 	= data['y']
z 	= data['z']
sourceterm = data['sourceterm']
# 
# 
# These are the plots.
d1 = np.real(phi)
d2 = np.real(sourceterm)
d3 = np.linalg.norm(np.real(E),axis=3)
d4 = np.real(E[:,:,:,0])
d5 = np.real(E[:,:,:,1])
d6 = np.real(E[:,:,:,2])
d7 = np.real(p_field_3d)
d8 = np.abs(p_field_3d)
d9 = np.linalg.norm(np.real(grad_p),axis=3)
#  
x_s = 25
x_e = 75
z_s = 25
z_e = 750
d1_min = np.amin(d1[x_s:x_e,x_s:x_e,z_s:z_e])
d1_max = np.amax(d1[x_s:x_e,x_s:x_e,z_s:z_e])
d2_min = np.amin(d2[x_s:x_e,x_s:x_e,z_s:z_e])
d2_max = np.amax(d2[x_s:x_e,x_s:x_e,z_s:z_e])
d3_min = np.amin(d3[x_s:x_e,x_s:x_e,z_s:z_e])
d3_max = np.amax(d3[x_s:x_e,x_s:x_e,z_s:z_e])
d4_min = np.amin(d4[x_s:x_e,x_s:x_e,z_s:z_e])
d4_max = np.amax(d4[x_s:x_e,x_s:x_e,z_s:z_e])
d5_min = np.amin(d5[x_s:x_e,x_s:x_e,z_s:z_e])
d5_max = np.amax(d5[x_s:x_e,x_s:x_e,z_s:z_e])
d6_min = np.amin(d6[x_s:x_e,x_s:x_e,z_s:z_e])
d6_max = np.amax(d6[x_s:x_e,x_s:x_e,z_s:z_e])
d7_min = np.amin(d7[x_s:x_e,x_s:x_e,z_s:z_e])
d7_max = np.amax(d7[x_s:x_e,x_s:x_e,z_s:z_e])
d8_min = np.amin(d8[x_s:x_e,x_s:x_e,z_s:z_e])
d8_max = np.amax(d8[x_s:x_e,x_s:x_e,z_s:z_e])
d9_min = np.amin(d9[x_s:x_e,x_s:x_e,z_s:z_e])
d9_max = np.amax(d9[x_s:x_e,x_s:x_e,z_s:z_e])


[resultx,resulty,resultz] = np.where(d3 == np.amax(d3[x_s:x_e,x_s:x_e,z_s:z_e] ) )
print('Mag Re E max ind:',resultx[0],resulty[0],resultz[0])
idx = resultz[0]  #  this is the starting index for the slider, the max. 
idx_xz = resulty[0]
[px,py,pz] = np.where(d8 == np.amax(d8[x_s:x_e,x_s:x_e,z_s:z_e] ) )
print('Abs Pressure max ind:',px[0],px[0],pz[0])
[px,py,pz] = np.where(d9 == np.amax(d9[x_s:x_e,x_s:x_e,z_s:z_e] ) )
print('Mag Re Grad Pressure max ind:',px[0],px[0],pz[0])


xy_extent_value = [p_x_scale[x_s],p_x_scale[x_e],p_x_scale[x_s],p_x_scale[x_e]]
xz_extent_value = [p_x_scale[x_s],p_x_scale[x_e],p_z_scale[z_s],p_z_scale[z_e]]


# figure axis setup 
fig, ax = plt.subplots(3,3,figsize=(7,7))
fig.subplots_adjust(bottom=0.15)
# 
im_h = ax[0,0].imshow(d1[:, :, idx], cmap='inferno', vmin=d1_min,vmax=d1_max,extent=xy_extent_value, interpolation='nearest')
divider = make_axes_locatable(ax[0,0])
cax = divider.append_axes("right", size="10%", pad=0.1)
cbar = plt.colorbar(im_h, cax=cax)
ax[0,0].title.set_text('Phi')

# 
im_i = ax[0,1].imshow(d2[:, :, idx], cmap='inferno', vmin=d2_min,vmax=d2_max, extent=xy_extent_value, interpolation='nearest')
divider = make_axes_locatable(ax[0,1])
cax = divider.append_axes("right", size="10%", pad=0.1)
cbar = plt.colorbar(im_i, cax=cax)
ax[0,1].title.set_text('Sourceterm')

# 
im_j = ax[0,2].imshow(d3[:, :, idx], cmap='inferno',vmin=d3_min,vmax=d3_max, extent=xy_extent_value, interpolation='nearest')
divider = make_axes_locatable(ax[0,2])
cax = divider.append_axes("right", size="10%", pad=0.1)
cbar = plt.colorbar(im_j, cax=cax)
# cbar.set_clim(-0.1, 0.1)
ax[0,2].title.set_text('Mag Re(AE)')

# 
im_k = ax[1,0].imshow(d4[:, :, idx], cmap='inferno', vmin=d4_min,vmax=d4_max, extent=xy_extent_value, interpolation='nearest')
divider = make_axes_locatable(ax[1,0])
cax = divider.append_axes("right", size="10%", pad=0.1)
cbar = plt.colorbar(im_k, cax=cax)
ax[1,0].title.set_text('AE_x')

im_l = ax[1,1].imshow(d5[:, :, idx], cmap='inferno', vmin=d5_min,vmax=d5_max, extent=xy_extent_value, interpolation='nearest')
divider = make_axes_locatable(ax[1,1])
cax = divider.append_axes("right", size="10%", pad=0.1)
cbar = plt.colorbar(im_l, cax=cax)
ax[1,1].title.set_text('AE_y')

im_m = ax[1,2].imshow(d6[:, :, idx], cmap='inferno', vmin=d6_min,vmax=d6_max, extent=xy_extent_value, interpolation='nearest')
divider = make_axes_locatable(ax[1,2])
cax = divider.append_axes("right", size="10%", pad=0.1)
cbar = plt.colorbar(im_m, cax=cax)
ax[1,2].title.set_text('AE_z')


im_n = ax[2,0].imshow(d7[:, :, idx], cmap='inferno', vmin=d7_min,vmax=d7_max, extent=xy_extent_value, interpolation='nearest')
divider = make_axes_locatable(ax[2,0])
cax = divider.append_axes("right", size="10%", pad=0.1)
cbar = plt.colorbar(im_n, cax=cax)
ax[2,0].title.set_text('Re(P)')

im_o = ax[2,1].imshow(d8[:, :, idx], cmap='inferno', vmin=d8_min,vmax=d8_max, extent=xy_extent_value, interpolation='nearest')
divider = make_axes_locatable(ax[2,1])
cax = divider.append_axes("right", size="10%", pad=0.1)
cbar = plt.colorbar(im_o, cax=cax)
ax[2,1].title.set_text('abs(P)')

im_p = ax[2,2].imshow(d9[:, :, idx], cmap='inferno', vmin=d9_min,vmax=d9_max, extent=xy_extent_value, interpolation='nearest')
divider = make_axes_locatable(ax[2,2])
cax = divider.append_axes("right", size="10%", pad=0.1)
cbar = plt.colorbar(im_p, cax=cax)
ax[2,2].title.set_text('Mag Re(grad P)')

# setup a slider axis and the Slider
ax_depth = plt.axes([0.23, 0.02, 0.56, 0.04])
slider_depth = Slider(ax_depth, 'slice', 0, d1.shape[2]-1, valinit=idx)
# update the figure with a change on the slider 
def update_depth(val):
    idx = int(round(slider_depth.val))
    im_h.set_data(d1[:, :, idx])
    im_i.set_data(d2[:, :, idx])  
    im_j.set_data(d3[:, :, idx])    
    im_k.set_data(d4[:, :, idx]) 
    im_l.set_data(d5[:, :, idx]) 
    im_m.set_data(d6[:, :, idx])     
    im_n.set_data(d7[:, :, idx]) 
    im_o.set_data(d8[:, :, idx])   
    im_p.set_data(d9[:, :, idx])   

# for x in ax.ravel():
#     x.axis("off")
slider_depth.on_changed(update_depth)
plt.show()


# Do it for XZ too. 
# figure axis setup 
fig, ax = plt.subplots(3,3,figsize=(7,7))
fig.subplots_adjust(bottom=0.15)
# 
# 
xmin = -2
xmax = 2
# The whole thing: 
# ymin = p_z_scale[z_s]
# ymax = p_z_scale[z_e]
# print (p_z_scale[z_s],p_z_scale[z_e])
ymin = 50
ymax = 55

im_h = ax[0,0].imshow(d1[:, idx_xz, :].T, cmap='inferno', vmin=d1_min,vmax=d1_max,extent=xz_extent_value, interpolation='nearest')
divider = make_axes_locatable(ax[0,0])
cax = divider.append_axes("right", size="10%", pad=0.1)
cbar = plt.colorbar(im_h, cax=cax)
ax[0,0].title.set_text('Phi')
ax[0,0].set_xlim([xmin,xmax])
ax[0,0].set_ylim([ymin,ymax])

# 
im_i = ax[0,1].imshow(d2[:, idx_xz, :].T, cmap='inferno', vmin=d2_min,vmax=d2_max, extent=xz_extent_value, interpolation='nearest')
divider = make_axes_locatable(ax[0,1])
cax = divider.append_axes("right", size="10%", pad=0.1)
cbar = plt.colorbar(im_i, cax=cax)
ax[0,1].title.set_text('Sourceterm')
ax[0,1].set_xlim([xmin,xmax])
ax[0,1].set_ylim([ymin,ymax])

# 
im_j = ax[0,2].imshow(d3[:, idx_xz, :].T, cmap='inferno',vmin=d3_min,vmax=d3_max, extent=xz_extent_value, interpolation='nearest')
divider = make_axes_locatable(ax[0,2])
cax = divider.append_axes("right", size="10%", pad=0.1)
cbar = plt.colorbar(im_j, cax=cax)
# cbar.set_clim(-0.1, 0.1)
ax[0,2].title.set_text('AE real magnitude')
ax[0,2].set_xlim([xmin,xmax])
ax[0,2].set_ylim([ymin,ymax])

# 
im_k = ax[1,0].imshow(d4[:, idx_xz, :].T, cmap='inferno', vmin=d4_min,vmax=d4_max, extent=xz_extent_value, interpolation='nearest')
divider = make_axes_locatable(ax[1,0])
cax = divider.append_axes("right", size="10%", pad=0.1)
cbar = plt.colorbar(im_k, cax=cax)
ax[1,0].title.set_text('AE_x')
ax[1,0].set_xlim([xmin,xmax])
ax[1,0].set_ylim([ymin,ymax])

im_l = ax[1,1].imshow(d5[:, idx_xz, :].T, cmap='inferno', vmin=d5_min,vmax=d5_max, extent=xz_extent_value, interpolation='nearest')
divider = make_axes_locatable(ax[1,1])
cax = divider.append_axes("right", size="10%", pad=0.1)
cbar = plt.colorbar(im_l, cax=cax)
ax[1,1].title.set_text('AE_y')
ax[1,1].set_xlim([xmin,xmax])
ax[1,1].set_ylim([ymin,ymax])

im_m = ax[1,2].imshow(d6[:, idx_xz, :].T, cmap='inferno', vmin=d6_min,vmax=d6_max, extent=xz_extent_value, interpolation='nearest')
divider = make_axes_locatable(ax[1,2])
cax = divider.append_axes("right", size="10%", pad=0.1)
cbar = plt.colorbar(im_m, cax=cax)
ax[1,2].title.set_text('AE_z')
ax[1,2].set_xlim([xmin,xmax])
ax[1,2].set_ylim([ymin,ymax])

im_n = ax[2,0].imshow(d7[:, idx_xz, :].T, cmap='inferno', vmin=d7_min,vmax=d7_max, extent=xz_extent_value, interpolation='nearest')
divider = make_axes_locatable(ax[2,0])
cax = divider.append_axes("right", size="10%", pad=0.1)
cbar = plt.colorbar(im_n, cax=cax)
ax[2,0].title.set_text('Re(P)')
ax[2,0].set_xlim([xmin,xmax])
ax[2,0].set_ylim([ymin,ymax])

im_o = ax[2,1].imshow(d8[:, idx_xz, :].T, cmap='inferno', vmin=d8_min,vmax=d8_max, extent=xz_extent_value, interpolation='nearest')
divider = make_axes_locatable(ax[2,1])
cax = divider.append_axes("right", size="10%", pad=0.1)
cbar = plt.colorbar(im_o, cax=cax)
ax[2,1].title.set_text('abs(P)')
ax[2,1].set_xlim([xmin,xmax])
ax[2,1].set_ylim([ymin,ymax])

im_p = ax[2,2].imshow(d9[:, idx_xz, :].T, cmap='inferno', vmin=d9_min,vmax=d9_max, extent=xz_extent_value, interpolation='nearest')
divider = make_axes_locatable(ax[2,2])
cax = divider.append_axes("right", size="10%", pad=0.1)
cbar = plt.colorbar(im_p, cax=cax)
ax[2,2].title.set_text('Mag Re(grad P)')
ax[2,2].set_xlim([xmin,xmax])
ax[2,2].set_ylim([ymin,ymax])

# setup a slider axis and the Slider
ax_depth = plt.axes([0.23, 0.02, 0.56, 0.04])
slider_depth = Slider(ax_depth, 'slice', 0, d1.shape[1]-1, valinit=idx_xz)
# update the figure with a change on the slider 
def update_depth(val):
    idx_xz = int(round(slider_depth.val))
    im_h.set_data(d1[:, idx_xz, :].T)
    im_i.set_data(d2[:, idx_xz, :].T)  
    im_j.set_data(d3[:, idx_xz, :].T)    
    im_k.set_data(d4[:, idx_xz, :].T) 
    im_l.set_data(d5[:, idx_xz, :].T) 
    im_m.set_data(d6[:, idx_xz, :].T)     
    im_n.set_data(d7[:, idx_xz, :].T) 
    im_o.set_data(d8[:, idx_xz, :].T)   
    im_p.set_data(d9[:, idx_xz, :].T)   

# for x in ax.ravel():
#     x.axis("off")
slider_depth.on_changed(update_depth)
plt.show()
