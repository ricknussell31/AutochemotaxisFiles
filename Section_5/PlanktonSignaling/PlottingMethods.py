from scipy import *
import scipy.sparse as sp
import numpy as np
from numpy.random import rand, uniform
from scipy import sparse
import copy

def PartProj(Swimmers,pos,meshsize):
    
    x = r_[0:Swimmers.L:1j*meshsize] # setup the spatial mesh. It is a long row vector
    # Create some local coordinates for the square domain.
    y = 1*x
    xm,ym = np.meshgrid(x,y)
    f = np.zeros((meshsize,meshsize))
    Std = Swimmers.depVar

    boundaryCutoff = 64*Std
    
    intDelta = int((meshsize)*8*np.sqrt(Std)/Swimmers.L + 0.5)
    
    for i in range(0,Swimmers.num):
        A, B, C, D = 0,0,0,0
        p = pos[i,:]
        
        centerX = int((meshsize-1)*p[0]/Swimmers.L+0.5)
        centerY = int((meshsize-1)*p[1]/Swimmers.L+0.5)
        
        lowerX      = max(0,centerX-intDelta)
        lowerXplus  = max(0,centerX-intDelta + (meshsize-1))
        lowerXminus = max(0,centerX-intDelta - (meshsize-1))
        upperX      = min(meshsize,centerX+intDelta)
        upperXplus  = min(meshsize,centerX+intDelta + (meshsize-1))
        upperXminus = min(meshsize,centerX+intDelta - (meshsize-1))
        lowerY      = max(0,centerY-intDelta)
        lowerYplus  = max(0,centerY-intDelta + (meshsize-1))
        lowerYminus = max(0,centerY-intDelta - (meshsize-1))
        upperY      = min(meshsize,centerY+intDelta)
        upperYplus  = min(meshsize,centerY+intDelta + (meshsize-1))
        upperYminus = min(meshsize,centerY+intDelta - (meshsize-1))
        
        sliceX = slice(lowerX,upperX+1)
        sliceY = slice(lowerY,upperY+1)
        
        f[sliceY,sliceX] = f[sliceY,sliceX] + (1/(4*pi*Std))*np.exp(-((xm[sliceY,sliceX]-p[0])**2+(ym[sliceY,sliceX]-p[1])**2)/4/Std)
        if ((p[0])**2<boundaryCutoff):
            sliceX = slice(lowerXplus,upperXplus+1)
            sliceY = slice(lowerY,upperY+1)
            f[sliceY,sliceX] = f[sliceY,sliceX] + (1/(4*pi*Std))*np.exp(-((xm[sliceY,sliceX]-p[0]-Swimmers.L)**2+(ym[sliceY,sliceX]-p[1])**2)/4/Std)
            A = 1
        if ((p[0]-Swimmers.L)**2<boundaryCutoff):
            sliceX = slice(lowerXminus,upperXminus+1)
            sliceY = slice(lowerY,upperY+1)
            f[sliceY,sliceX] = f[sliceY,sliceX] + (1/(4*pi*Std))*np.exp(-((xm[sliceY,sliceX]-p[0]+Swimmers.L)**2+(ym[sliceY,sliceX]-p[1])**2)/4/Std)
            B = 1
        if ((p[1])**2<boundaryCutoff):
            sliceX = slice(lowerX,upperX+1)
            sliceY = slice(lowerYplus,upperYplus+1)
            f[sliceY,sliceX] = f[sliceY,sliceX] + (1/(4*pi*Std))*np.exp(-((xm[sliceY,sliceX]-p[0])**2+(ym[sliceY,sliceX]-p[1]-Swimmers.L)**2)/4/Std)
            C = 1
        if ((p[1]-Swimmers.L)**2<boundaryCutoff):
            sliceX = slice(lowerX,upperX+1)
            sliceY = slice(lowerYminus,upperYminus+1)
            f[sliceY,sliceX] = f[sliceY,sliceX] + (1/(4*pi*Std))*np.exp(-((xm[sliceY,sliceX]-p[0])**2+(ym[sliceY,sliceX]-p[1]+Swimmers.L)**2)/4/Std)
            D = 1
        if (A == 1 and C == 1): #Plankton in Lower Left Corner
            sliceX = slice(lowerXplus,upperXplus+1)
            sliceY = slice(lowerYplus,upperYplus+1)
            f[sliceY,sliceX] = f[sliceY,sliceX] + (1/(4*pi*Std))*np.exp(-((xm[sliceY,sliceX]-p[0]-Swimmers.L)**2+(ym[sliceY,sliceX]-p[1]-Swimmers.L)**2)/4/Std)
        if (A == 1 and D == 1): #Plankton in Lower Left Corner
            sliceX = slice(lowerXplus,upperXplus+1)
            sliceY = slice(lowerYminus,upperYminus+1)
            f[sliceY,sliceX] = f[sliceY,sliceX] + (1/(4*pi*Std))*np.exp(-((xm[sliceY,sliceX]-p[0]-Swimmers.L)**2+(ym[sliceY,sliceX]-p[1]+Swimmers.L)**2)/4/Std)
        if (B == 1 and C == 1): #Plankton in Upper Right Corner
            sliceX = slice(lowerXminus,upperXminus+1)
            sliceY = slice(lowerYplus,upperYplus+1)
            f[sliceY,sliceX] = f[sliceY,sliceX] + (1/(4*pi*Std))*np.exp(-((xm[sliceY,sliceX]-p[0]+Swimmers.L)**2+(ym[sliceY,sliceX]-p[1]-Swimmers.L)**2)/4/Std)
        if (B == 1 and D == 1): #Plankton in Lower Right Corner
            sliceX = slice(lowerXminus,upperXminus+1)
            sliceY = slice(lowerYminus,upperYminus+1)
            f[sliceY,sliceX] = f[sliceY,sliceX] + (1/(4*pi*Std))*np.exp(-((xm[sliceY,sliceX]-p[0]+Swimmers.L)**2+(ym[sliceY,sliceX]-p[1]+Swimmers.L)**2)/4/Std)
    return xm,ym,f*(Swimmers.density*Swimmers.L**2/Swimmers.num)


def BuildPeriodic(Swimmers,Mat):
    mgf                   = copy.deepcopy(Mat)
    s                     = mgf.reshape((Swimmers.N,Swimmers.N))
    sp                    = Swimmers.scalar_periodic.reshape((Swimmers.N+1,Swimmers.N+1))
    sp[0:Swimmers.N,0:Swimmers.N] = s[:,:]
    sp[Swimmers.N,0:Swimmers.N]   = sp[0,0:Swimmers.N]
    sp[0:Swimmers.N,Swimmers.N]   = sp[0:Swimmers.N,0]
    sp[Swimmers.N,Swimmers.N]     = sp[0,0]
    sp  = sp.reshape(((Swimmers.N+1),(Swimmers.N+1)))
    return(mgf)