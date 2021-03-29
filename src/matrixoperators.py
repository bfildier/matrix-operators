"""
Python module to collect useful matrix operations in atmospheric science 
(derivatives, filtering, integrals, divergence, rotationals, etc.).

Author: B. Fildier, 2021
"""

import numpy as np
from scipy.ndimage import gaussian_filter,gaussian_filter1d


class MatrixOperators():
    
    def derivative(self,a,x,axis=0,order=1):
        """Derivative da\dx, d2a/dx2, etc. along given axis"""
        
        if order == 1: # df/dx
            
            # dimensions
            ash = a.shape
            
            ##-- compute first order derivative
            if len(ash) == 1: # if vector

                # First order difference, truncating edges
                def diff1D(vec,coord):
                    return np.convolve(vec,[1,-1],mode='valid')/np.convolve(coord,[1,-1],mode='valid')

                # Interpolate on initial grid and append nans
                def regrid(vec):
                    return np.hstack([[np.nan],np.convolve(vec,[0.5,0.5],mode='valid'),[np.nan]])

                da_dx_mids = diff1D(a,x)
                da_dx = regrid(da_dx_mids)

            else: # if matrix

                Nx = ash[axis]
            
                ##-- first order derivative
                # duplicate z values to arr shape
                x_full = np.moveaxis(np.tile(x,(*ash[:axis],*ash[axis+1:],1)),-1,axis)
                # derivative
                da = np.take(a,range(1,Nx),axis=axis)-np.take(a,range(0,Nx-1),axis=axis)
                dx = np.take(x_full,range(1,Nx),axis=axis)-np.take(x_full,range(0,Nx-1),axis=axis)
                da_dx_mids = da/dx

                ##-- regrid and append nans on both sides
                da_dx_grid = 0.5*(np.take(da_dx_mids,range(1,Nx-1),axis=axis)+\
                                  np.take(da_dx_mids,range(0,Nx-2),axis=axis))
                # append nans
                hyperspace_nans = np.nan*np.zeros((*ash[:axis],1,*ash[axis+1:]))
                da_dx = np.concatenate([hyperspace_nans,da_dx_grid,hyperspace_nans],axis=axis)
            
            return da_dx, da_dx_mids


    def gaussianFilter(self,a,sigma,axis=0,**kwargs):
        """Gaussian filter in 1 or 2 dimensions"""
        
        if isinstance(axis,int):
            
            ## 1D, use gaussian_filter1d
            return gaussian_filter1d(a,sigma=sigma,axis=axis,mode='constant')    
            
        elif isinstance(axis,list):
            
            ## ND, use multidimensional gaussian_filter
            """Smooth in x an y and recombine with same shape.
            Assumes dimensions T,Z,Y,X"""
            
            ashape = a.shape
            
            if len(ashape) == 3:
                Nt,Ny,Nx = ashape
                Nz = 0
            elif len(ashape) == 4:
                Nt,Nz,Ny,Nx = ashape
            
            a_out = np.nan*np.zeros(ashape)
            
            for i_t in range(Nt):
                
                if Nz == 0:
                    a_out[i_t,:,:] = gaussian_filter(a[i_t,:,:],sigma=sigma,mode='wrap')
                
                else:
                    for i_z in range(Nz):
        
                        a_out[i_t,i_z,:,:] = gaussian_filter(a[i_t,i_z,:,:],sigma=sigma,mode='wrap')
                    
            return a_out


            
            

            
            
        