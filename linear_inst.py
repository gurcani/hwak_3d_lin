#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 22 12:45:37 2022

@author: ogurcan
"""

import numpy as np
import matplotlib.pylab as plt

def oneover(x):
    res=np.zeros_like(x)
    inds=np.nonzero(x)
    res[inds]=1/x[inds]
    return res

def init_kspace_grid(Nx,Ny,Nz,Lx,Ly,Lz):
    dkx=2*np.pi/Lx
    dky=2*np.pi/Ly
    dkz=2*np.pi/Lz
    kxl=np.r_[0:int(Nx/2),-int(Nx/2):0]*dkx
    kyl=np.r_[0:int(Ny/2),-int(Ny/2):0]*dky
    kzl=np.r_[0:int(Nz/2+1)]*dkz
    kx,ky,kz=np.meshgrid(kxl,kyl,kzl,indexing='ij')
    return kx,ky,kz

def init_linmats(pars,kx,ky,kz,ksqr):
    #Initializing the linear matrices
    C,kap,nu,D,nuZ,DZ=[pars[l] for l in ['C','kap','nu','D','nuZ','DZ']]
    nuksqrpow=pars['nuksqrpow']
    lm=np.zeros((2,2)+kx.shape,dtype=complex)
    lm[0,0,]=-C*kz**2*oneover(ksqr)-nu*ksqr**nuksqrpow-nuZ*kz**2*nuksqrpow
    lm[0,1,]=C*kz**2*oneover(ksqr)
    lm[1,0,]=-1j*kap*ky+C*kz**2
    lm[1,1,]=-C*kz**2-D*ksqr**nuksqrpow-DZ*kz**2*nuksqrpow
    lm[:,:,0,0,]=0.0
    return lm

def lincompfreq(lm):
    w,v=np.linalg.eig(lm.T)
    ia=np.argsort(w.real,axis=-1)
    lam=np.take_along_axis(w, np.flip(ia,axis=-1), axis=-1).T
    vi=np.zeros_like(v.T)
    vi[0,]=np.take_along_axis(v[:,:,:,0,:], np.flip(ia,axis=-1), axis=-1).T
    vi[1,]=np.take_along_axis(v[:,:,:,1,:], np.flip(ia,axis=-1), axis=-1).T
    return lam,vi #,vi

Nx,Ny,Nz=256,256,256
Lx,Ly,Lz=32*np.pi,32*np.pi,32*np.pi
kx,ky,kz=init_kspace_grid(Nx,Ny,Nz,Lx,Ly,Lz)
ksqr=kx**2+ky**2

pars={'C':1.0,
      'kap':1.0,
      'nu':5e-3,
      'D':0.0,
      'nuZ':5e-3,
      'DZ':0.0,
      'nuksqrpow':1}

nu=pars['nu']

#kx,ky,kz=np.fft.fftshift(kx,axes=(0,1)),np.fft.fftshift(ky,axes=(0,1)),np.fft.fftshift(kz,axes=(0,1))
lm=init_linmats(pars,kx,ky,kz,ksqr)
lam,vi=lincompfreq(lm)
om=1j*lam

plt.figure()
slz=slice(None,None,int(Nz/16))
plt.plot(ky[0,:int(Ny/2),slz],om[0,0,:int(Ny/2),slz].imag,'x')
plt.plot(ky[0,:int(Ny/2),0],-nu*ky[0,:int(Ny/2),0]**2,'k--')
plt.legend(['$k_z='+str(l)+'$' for l in kz[0,0,slz]]+['$-nu*k_y^2$'])
plt.xlabel('$k_y$')
plt.ylabel('$\gamma(k_y) = \gamma(k_x=0,k_y,k_z=k_{zi})$')

plt.figure()
sly=slice(None,int(Ny/2),int(Ny/16))
plt.plot(kz[0,sly,:].T,om[0,0,sly,:].T.imag,'x')
plt.plot(kz[0,0,:],-nu*kz[0,0,:]**2,'k--')
plt.legend(['$k_y='+str(l)+'$' for l in ky[0,sly,0]]+['$-nu*k_z^2$'])
plt.xlabel('$k_z$')
plt.ylabel('$\gamma(k_z) = \gamma(k_x=0,k_y=k_{yi},k_z)$')

plt.figure()
plt.pcolormesh(np.fft.fftshift(ky[0,],axes=0),np.fft.fftshift(kz[0,],axes=0),
               np.fft.fftshift(om[0,0,:,:].imag,axes=0),cmap='seismic',vmax=0.2,vmin=-0.2,rasterized=True)
plt.xlabel('$k_y$')
plt.ylabel('$k_z$')
plt.title('$\gamma(k_y,k_z) = \gamma(k_x=0,k_y,k_z)$')
plt.colorbar()

gam0=np.max(om.imag[0,])
ia=np.nonzero(gam0==om.imag[0,])
iaz=ia[2][0]

plt.figure()
plt.pcolormesh(np.fft.fftshift(kx[:,:int(Ny/2),iaz],axes=0),np.fft.fftshift(ky[:,:int(Ny/2),iaz],axes=0),
               np.fft.fftshift(om[0,:,:int(Ny/2),iaz].imag,axes=0),cmap='seismic',vmax=0.2,vmin=-0.2,rasterized=True)
plt.xlabel('$k_x$')
plt.ylabel('$k_y$')
plt.title('$\gamma(k_y,k_z) = \gamma(k_x,k_y,k_z=k_{zmax})$')
plt.colorbar()
plt.show()
