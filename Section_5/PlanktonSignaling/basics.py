# coding: utf-8
from scipy import *
from math import acos, asin
import scipy.sparse as sp
import numpy as np
from scipy.interpolate import RectBivariateSpline,griddata
from scipy.sparse.linalg import spsolve
from numpy.random import rand, uniform
from scipy import sparse

class Background_Field(object):
    "A class that creates the background concentration field and evolves"
    
    # class builder initiation 
    def __init__(self,N=30,L=10,d1=0.1,d2=0.1,k=0.1,*args,**kwargs):
                
        self.N       = N # The number of mesh points
        self.k       = k 
        self.L       = L

        self.d1 = d1
        self.d2 = d2
        
        self.x = r_[0:self.L:self.L/self.N]
        self.h = self.x[1]-self.x[0] # spatial mesh size

        self.x_periodic = np.append(self.x,self.L)
        self.y_periodic = 1*self.x_periodic

        # Create some local coordinates for the square domain.
        self.y = 1*self.x
        self.xm,self.ym = np.meshgrid(self.x,self.y)

        self.xm_periodic,self.ym_periodic = np.meshgrid(self.x_periodic,self.y_periodic)
        
        self.scalar = self.x
        for counter in range(0,self.N-1):
            self.scalar = np.append(self.scalar,self.x)

        self.scalar_periodic = self.x_periodic
        for counter in range(0,self.N):
            self.scalar_periodic = np.append(self.scalar_periodic,self.x_periodic)        
        
        self.SetAlpha()
        self.BuildMatrixA1()
        self.BuildMatrixA2()
        self.BuildMatrices() # build M1 and M2
    
    # Forward one time step. Just diffuse
    def Update(self, vectors): # Remark: we feed in all info, but only those flagged
        # will be emitting substances to change the background field
        
        self.scalar = spsolve(self.M1, self.M2.dot(vectors))

        return self.scalar
    
    # Set up initial condition
    def SetIC(self,f):
        ic = f(self.xm,self.ym)
        ic = ic.reshape((self.N**2,))
        self.IC = ic
        self.scalar = ic
        
    def Meshed(self):
        return(np.roll(self.scalar.reshape((self.N,self.N)),int()))

    def BuildPeriodic(self):
        s                     = self.scalar.reshape((self.N,self.N))
        sp                    = self.scalar_periodic.reshape((self.N+1,self.N+1))
        sp[0:self.N,0:self.N] = s[:,:]
        sp[self.N,0:self.N]   = sp[0,0:self.N]
        sp[0:self.N,self.N]   = sp[0:self.N,0]
        sp[self.N,self.N]     = sp[0,0]
        self.scalar_periodic  = self.scalar_periodic.reshape(((self.N+1),(self.N+1)))
    
    # Compute alpha
    def SetAlpha(self):
        self.alpha = self.d1*self.k/(self.h)**2
        
    # Build the N x N matrix A1 for 1-Dimensional Crank-Nicoleson Method
    def BuildMatrixA1(self):
        diag = ones(self.N)*(1+4*self.alpha/2+self.k*self.d2/2)
        data = np.array([-ones(self.N)*self.alpha/2,-ones(self.N)*self.alpha/2,
                         diag, -ones(self.N)*self.alpha/2,-ones(self.N)*self.alpha/2])
        #off-diag and corners are -alpha
        self.A1 = sp.spdiags(data,[1-self.N,-1,0,1,self.N-1],self.N,self.N)
        
    def BuildMatrixA2(self):
        diag = ones(self.N)*(1-4*self.alpha/2-self.k*self.d2/2)
        data = np.array([ones(self.N)*self.alpha/2, ones(self.N)*self.alpha/2,
                         diag, ones(self.N)*self.alpha/2, ones(self.N)*self.alpha/2])
        #off-diag and corners are alpha
        self.A2 = sp.spdiags(data,[1-self.N,-1,0,1,self.N-1],self.N,self.N)
    
    # Build the big matrices M1 M2 using I, A1 and A2
    def BuildMatrices(self):
        ############ Build M1
        self.I = sp.identity(self.N) # Identity N x N Sparse Matrix
        self.E = sp.csr_matrix((self.N,self.N)) # Zero N x N Sparse Matrix
        Rows = {i: self.E for i in range(self.N)} # Empty rows of tile matrices
        Rows[0] = self.A1
        Rows[1] = -self.I*self.alpha/2
        Rows[self.N-1] = -self.I*self.alpha/2
        # Creating rows
        for i in range(self.N):
            for j in range(1,self.N):
                if j == i:
                    buildblock = self.A1
                elif j == (i-1 % self.N) or j == (i+1 % self.N) or (j==self.N-1 and i==0):
                    # a cheap way to fix
                    buildblock = -self.I*self.alpha/2
                else:
                    buildblock = self.E
                Rows[i] = sp.hstack([Rows[i],buildblock]) # Stack matrices horizontally to create rows
                
        # Stack rows together vertically to get M1
        self.M1 = Rows[0]
        for i in range(1,self.N):
            self.M1 = sp.vstack([self.M1,Rows[i]])
        self.M1 = self.M1.tocsr()    
        ############ Build M2
        Rows = {i: self.E for i in range(self.N)} # Empty rows of tile matrices
        Rows[0] = self.A2
        Rows[1] = self.I*self.alpha/2
        Rows[self.N-1] = self.I*self.alpha/2
        # Creating rows
        for i in range(self.N):
            for j in range(1,self.N):
                if j == i:
                    buildblock = self.A2
                elif j == (i-1 % self.N) or j == (i+1 % self.N) or (j==self.N-1 and i==0):
                    # a cheap way to fix
                    buildblock = self.I*self.alpha/2
                else:
                    buildblock = self.E
                Rows[i] = sp.hstack([Rows[i],buildblock]) # Stack matrices horizontally to create rows
                
        # Stack rows together vertically to get M2
        self.M2 = Rows[0]
        for i in range(1,self.N):
            self.M2 = sp.vstack([self.M2,Rows[i]])
        self.M2 = self.M2.tocsr()
        
    def CheckM():
        checkmatrix = self.M1.toarray()
        print(np.array2string(checkmatrix,precision=2))
        checkmatrix = self.M2.toarray()
        print(np.array2string(checkmatrix,precision=2))
        
    def CheckA():
        checkmatrix = self.A1.toarray()
        print(np.array2string(checkmatrix,precision=2))
        checkmatrix = self.A2.toarray()
        print(np.array2string(checkmatrix,precision=2))


class Plankton(Background_Field):
    
    def __init__(self,depFcn,d1=0.1,d2=0.1,k=0.02,Const = 3,depMaxStr=1.0e-10,delta=1.0e-8,
                 depTransWidth=0.001,depThreshold=0.008,num = 400,c0=0.012,*args,**kwargs):
        
        self.k = k #k is delta t
        self.d1 = d1
        self.d2 = d2
        self.depVar = Const*self.k*self.d1       #Deposition variable (Gaussian deposition)

        self.depMaxStr = depMaxStr #Deposition maximum strength
        self.depFcn = depFcn
        self.depTransWidth = depTransWidth
        self.depThreshold = depThreshold
        self.num = num
        self.c0 = c0 #Initial Chemical Steady State
        self.args = args
        self.kwargs = kwargs
        
        self.delta = delta

        super(Plankton,self).__init__(d1=self.d1, d2=self.d2,k=self.k,*args,**kwargs)
        self.density = self.d2*self.c0/self.depFcn(self.c0,self.depMaxStr,self.depThreshold,self.depTransWidth)
            
    def RT(self,pos,vel,c,grad_c):
        # Actually, I need to do this as tumble and run, TR.
        for j in range(0,len(pos)):
            alpha = 1/np.sqrt(self.delta**2 + dot(grad_c[j],vel[j]))
            if (rand() < self.k*0.5*(1-alpha*dot(vel[j],grad_c[j]))):
                th = rand()*2*pi
                vel[j] = array([cos(th),sin(th)])
        for j in range(0,len(pos)):
            pos[j] += self.k*vel[j]
            pos[j] = mod(pos[j],self.L)
            
    def Update(self,vectors,pos,vel):
        PlankDensity = self.density*self.L**2/self.num
        c      = self.scalarInterp(pos)
        grad_c = self.scalarGrad(pos)
        self.RT(pos,vel,c,grad_c)
        
        depStr = self.depFcn(c,self.depMaxStr,self.depThreshold,self.depTransWidth,*self.args,**self.kwargs)
        f = zeros((self.N,self.N))
        for p,str in zip(pos,depStr):
            
            A = 0
            B = 0
            C = 0
            D = 0
            
            f = f + str*exp(-((self.xm-p[0])**2+(self.ym-p[1])**2)/4/self.depVar)/(4*pi*self.depVar)
            # We must be cautious about Periodic BCs
            # This code includes the previously missing diagonal element
            # Works for all domains [0,L] x [0,L]
            if ((p[0])**2<64*self.depVar):
                f = f + str*exp(-((self.xm-p[0]-self.L)**2+(self.ym-p[1])**2)/4/self.depVar)/(4*pi*self.depVar)
                A = 1
                
            if ((p[0]-self.L)**2<64*self.depVar):
                f = f + str*exp(-((self.xm-p[0]+self.L)**2+(self.ym-p[1])**2)/4/self.depVar)/(4*pi*self.depVar)
                B = 1
                
            if ((p[1])**2<64*self.depVar):
                f = f + str*exp(-((self.xm-p[0])**2+(self.ym-p[1]-self.L)**2)/4/self.depVar)/(4*pi*self.depVar)
                C = 1
                
            if ((p[1]-self.L)**2<64*self.depVar):
                f = f + str*exp(-((self.xm-p[0])**2+(self.ym-p[1]+self.L)**2)/4/self.depVar)/(4*pi*self.depVar)
                D = 1
                
            if (A == 1 and C == 1): #Plankton in Lower Left Corner
                f = f + str*exp(-((self.xm-p[0]-self.L)**2+(self.ym-p[1]-self.L)**2)/4/self.depVar)/(4*pi*self.depVar)
            if (A == 1 and D == 1): #Plankton in Lower Left Corner
                f = f + str*exp(-((self.xm-p[0]-self.L)**2+(self.ym-p[1]+self.L)**2)/4/self.depVar)/(4*pi*self.depVar)
            if (B == 1 and C == 1): #Plankton in Upper Right Corner
                f = f + str*exp(-((self.xm-p[0]+self.L)**2+(self.ym-p[1]-self.L)**2)/4/self.depVar)/(4*pi*self.depVar)
            if (B == 1 and D == 1): #Plankton in Lower Right Corner
                f = f + str*exp(-((self.xm-p[0]+self.L)**2+(self.ym-p[1]+self.L)**2)/4/self.depVar)/(4*pi*self.depVar)
                
        f = f.reshape((self.N*self.N,))
        self.scalar = spsolve(self.M1, self.M2.dot(vectors)+self.k*(PlankDensity)*f)
        return(self.scalar)
    
    def UpdateSlice(self, vectors, pos, vel):
        PlankDensity = self.density*self.L**2/self.num
        c      = self.scalarInterp(pos)
        grad_c = self.scalarGrad(pos)
        self.RT(pos,vel,c,grad_c)

        Std = self.depVar
        boundaryCutoff = 64*Std
        intDelta = int((self.N-1)*8*sqrt(Std)/self.L+0.5)
        f = zeros((self.N,self.N))
        meshsize = self.N
        depStr = self.depFcn(c,self.depMaxStr,self.depThreshold,self.depTransWidth,*self.args,**self.kwargs)

        for p,str in zip(pos,depStr):

            A, B, C, D = 0,0,0,0

            centerX = int((meshsize-1)*p[0]/self.L+0.5)
            centerY = int((meshsize-1)*p[1]/self.L+0.5)

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
            f[sliceY,sliceX] = f[sliceY,sliceX] + str*(1/(4*pi*Std))*exp(-((self.xm[sliceY,sliceX]-p[0])**2+(self.ym[sliceY,sliceX]-p[1])**2)/4/Std)
            if ((p[0])**2<boundaryCutoff):
                sliceX = slice(lowerXplus,upperXplus+1)
                sliceY = slice(lowerY,upperY+1)
                f[sliceY,sliceX] = f[sliceY,sliceX] + str*(1/(4*pi*Std))*exp(-((self.xm[sliceY,sliceX]-p[0]-self.L)**2+(self.ym[sliceY,sliceX]-p[1])**2)/4/Std)
                A = 1
            if ((p[0]-self.L)**2<boundaryCutoff):
                sliceX = slice(lowerXminus,upperXminus+1)
                sliceY = slice(lowerY,upperY+1)
                f[sliceY,sliceX] = f[sliceY,sliceX] + str*(1/(4*pi*Std))*exp(-((self.xm[sliceY,sliceX]-p[0]+self.L)**2+(self.ym[sliceY,sliceX]-p[1])**2)/4/Std)
                B = 1
            if ((p[1])**2<boundaryCutoff):
                sliceX = slice(lowerX,upperX+1)
                sliceY = slice(lowerYplus,upperYplus+1)
                f[sliceY,sliceX] = f[sliceY,sliceX] + str*(1/(4*pi*Std))*exp(-((self.xm[sliceY,sliceX]-p[0])**2+(self.ym[sliceY,sliceX]-p[1]-self.L)**2)/4/Std)
                C = 1
            if ((p[1]-self.L)**2<boundaryCutoff):
                sliceX = slice(lowerX,upperX+1)
                sliceY = slice(lowerYminus,upperYminus+1)
                f[sliceY,sliceX] = f[sliceY,sliceX] + str*(1/(4*pi*Std))*exp(-((self.xm[sliceY,sliceX]-p[0])**2+(self.ym[sliceY,sliceX]-p[1]+self.L)**2)/4/Std)
                D = 1
            if (A == 1 and C == 1): #Plankton in Lower Left Corner
                sliceX = slice(lowerXplus,upperXplus+1)
                sliceY = slice(lowerYplus,upperYplus+1)
                f[sliceY,sliceX] = f[sliceY,sliceX] + str*(1/(4*pi*Std))*exp(-((self.xm[sliceY,sliceX]-p[0]-self.L)**2+(self.ym[sliceY,sliceX]-p[1]-self.L)**2)/4/Std)
            if (A == 1 and D == 1): #Plankton in Lower Left Corner
                sliceX = slice(lowerXplus,upperXplus+1)
                sliceY = slice(lowerYminus,upperYminus+1)
                f[sliceY,sliceX] = f[sliceY,sliceX] + str*(1/(4*pi*Std))*exp(-((self.xm[sliceY,sliceX]-p[0]-self.L)**2+(self.ym[sliceY,sliceX]-p[1]+self.L)**2)/4/Std)
            if (B == 1 and C == 1): #Plankton in Upper Right Corner
                sliceX = slice(lowerXminus,upperXminus+1)
                sliceY = slice(lowerYplus,upperYplus+1)
                f[sliceY,sliceX] = f[sliceY,sliceX] + str*(1/(4*pi*Std))*exp(-((self.xm[sliceY,sliceX]-p[0]+self.L)**2+(self.ym[sliceY,sliceX]-p[1]-self.L)**2)/4/Std)
            if (B == 1 and D == 1): #Plankton in Lower Right Corner
                sliceX = slice(lowerXminus,upperXminus+1)
                sliceY = slice(lowerYminus,upperYminus+1)
                f[sliceY,sliceX] = f[sliceY,sliceX] + str*(1/(4*pi*Std))*exp(-((self.xm[sliceY,sliceX]-p[0]+self.L)**2+(self.ym[sliceY,sliceX]-p[1]+self.L)**2)/4/Std)
        f = f.reshape((self.N*self.N,))
        self.scalar = spsolve(self.M1, self.M2.dot(vectors)+self.k*(PlankDensity)*f)
        return(self.scalar)

    def scalarInterp(self,pos):
        self.BuildPeriodic()
        bspline = RectBivariateSpline(self.x_periodic,self.y_periodic,self.scalar_periodic)
        
        # Flipping x and y inputs because self.Meshed() has reversed row
        # major formatting which goes back to meshgrid which goes back to
        # Matlab's old mistake.
        
        # The alternative is to transpose Meshed.
        
        return(bspline.ev(pos[:,1],pos[:,0]))

    # Assumes a [0,L]x[0,L] domain.
    #Fixing Periodic BCs
    def scalarGrad(self,xp,dx=1.0e-4):
        dx = dx*self.L
        self.BuildPeriodic()
        bspline = RectBivariateSpline(self.x_periodic,self.y_periodic,self.scalar_periodic)
        p       = array([mod(xp + array([dx,0]),self.L),mod(xp - array([dx,0]),self.L),mod(xp + array([0,dx]),self.L),
                                      mod(xp - array([0,dx]),self.L)])

        # Flipping x and y inputs because self.Meshed() has reversed row
        # major formatting which goes back to meshgrid which goes back to
        # Matlab's old mistake.
        
        # The alternative is to transpose Meshed.

        dp = bspline.ev(p[:,:,1],p[:,:,0])
        
        diffs = array([dp[0]-dp[1],dp[2]-dp[3]])/2/dx
        diffs = diffs.T
        
        return(diffs)

    def ORIGscalarInterp(self,p):
        return(griddata((self.xm.reshape(self.N**2,),self.ym.reshape(self.N**2,)),self.scalar,p,method='cubic'))

    # Assumes a [0,L]x[0,L] domain.
    def ORIGscalarGrad(self,xp,dx=1.0e-4):
        dx = self.L*dx
        dp = array(self.scalarInterp([mod(xp + array([dx,0]),self.L),mod(xp - array([dx,0]),self.L),mod(xp + array([0,dx]),self.L),
                                      mod(xp - array([0,dx]),self.L)]))
        diffs = array([dp[0]-dp[1],dp[2]-dp[3]])/2/dx
        diffs = diffs.T
        return(diffs)
        
    def MatrixBuildStable(self,k1,k2,s,d3):
        #Builds the fourier transform matrix that is used in the stability calculations
        kappa = k1 + 1j*k2
        kappab = k1 - 1j*k2
        psib = self.density/(2*pi)
        Z = zeros((2*s+1,1),dtype=complex)
        SupD = ones(2*s,dtype=complex)*(1j/2)*(kappa)
        SubD = ones(2*s,dtype=complex)*(1j/2)*(kappab)
        MD = zeros(2*s+1,dtype=complex) + -1/2
        MD[s]=0
        N = sparse.diags([MD,SupD,SubD],[0,1,-1])
        Z[s-1]=-psib*kappa*1j/(4*self.delta)
        Z[s+1]=-psib*kappab*1j/(4*self.delta)
        F = sparse.hstack([sparse.bsr_matrix(Z),N])
        M = zeros((1,2*s+2),dtype=complex)
        M[0,0] = d3 - self.d1*(k1**2 + k2**2)
        M[0,s+1] = 2*pi*self.depFcn(self.c0,self.depMaxStr,self.depThreshold,self.depTransWidth)
        return(sparse.vstack([sparse.bsr_matrix(M),F]))
    
    def CheckStability(self):
        #Calculuates the most unstable wave number in the system
        f0 = self.depFcn(self.c0,self.depMaxStr,self.depThreshold,self.depTransWidth)
        f1 = self.depFcn(self.c0+0.00001,self.depMaxStr,self.depThreshold,self.depTransWidth)
        f2 = self.depFcn(self.c0-0.00001,self.depMaxStr,self.depThreshold,self.depTransWidth)
        d1 = self.d1
        d2 = self.d2        
        fp = (f1 - f2)/(2*0.0001)
        d3 = self.density*fp - d2
        eigs = []
        modes = 100
        #k11 = linspace(0,self.L/2,100)
        k11 = linspace(0,10,400)
        for j in k11:
            try:
                eigs=np.append(eigs,sparse.linalg.eigs(self.MatrixBuildStable(j,j,modes,d3),which='LR',k=1)[0][0])
            except:
                eigs = np.append(eigs,max(real(sparse.linalg.eigs(self.MatrixBuildStable(j,j,modes,d3))[0])))
        normMax = round(sqrt(2)*k11[argmax(eigs)],3)
        if (normMax > self.L/(2*pi)):
            print('These parameter values will make the system unstable.')
        else:
            print('These parameter values will make the system stable.')
        print('Norm of most unstable wave number: {0}'.format(normMax))


