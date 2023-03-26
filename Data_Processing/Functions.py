import numpy as np
import math
import sys
import pdb
#pdb.set_trace() 
from numpy import linalg as LA
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from scipy import integrate
from scipy import sparse
from scipy.sparse.linalg import spsolve
from numpy import diff
import scipy
import scipy.stats as ss
from scipy.signal import find_peaks
from scipy.stats import iqr
from scipy.stats import skew, kurtosis
from scipy import signal



def read_samp_freq(file):
    file = open(file)
    X=[]
    for line in file:               
        n=line.split(' ')
        n[-1]=n[-1].replace('\n','')
        X.append(n[0].replace('\t',''))
    return float(X[8])


def read_data(file):
    file = open(file)
    X=[]
    for line in file:               
        n=line.split(',')
        n[-1]=n[-1].replace('\n','')
        if n[0]!= '\x1a':
            X.append(n[0])
    X=np.array(X).astype(float)
    return X

def read_data_NI(file):
    file = open(file)
    X=[]
    for line in file:               
        n=line.split('\t')
        n[-1]=n[-1].replace('\n','')
        for i in range(0,len(n)):
            if 'u' in n[i]:
                n[i]=float(n[i].replace('u',''))/1000
            elif 'm' in n[i]:
                n[i]=float(n[i].replace('m',''))
            elif 'n' in n[i]:
                n[i]=float(n[i].replace('n','')) /1000000
            else:
                n[i]=float(n[i])           
        X.append(n)
    X=np.array(X).astype(float)
    return X


def read_data_NI2(file):
    file = open(file)
    X=[]
    store_data=False
    for line in file:               
        n=line.split('\t')
        n[-1]=n[-1].replace('\n','')
        if '' in n:             
            n.remove('')
        if n!=[]:
            if n[0]=='Delta_X':
                dt=float(n[-2])
            if store_data:
                X.append(n)        
            if n[-1]=='Comment':
                store_data=True             
    X=np.array(X).astype(float)
    return [X,dt]


def baseline_als(y, lam, p, niter=10):
    L = len(y)
    D = sparse.diags([1,-2,1],[0,-1,-2], shape=(L,L-2))
    w = np.ones(L)
    for i in range(niter):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + lam * D.dot(D.transpose())
        z = spsolve(Z, w*y)
        w = p * (y > z) + (1-p) * (y < z)
    return z

def integrate_f(y,dt):
    area=[0]
    for i in range(1,len(y)):
        area_i=dt*(y[i]+y[i-1])/2
        area.append(area[-1]+area_i)
        #pdb.set_trace() 
    return np.array(area)


def derivative_f(y,dt):
    dy = diff(y)/dt
    return dy



def correlation_test(x,y,l=0.5):
    Autocorr=all(x==y)
    y=y-np.mean(y)
    phi_j=[]
    N=len(y)
    n=int(l*N)
    if Autocorr:
        for j in range(0,n):
            #y_sq=y[0:N-j]**2
            #y_j=y[0+j:N]
            y_sq=y[0:n]**2
            y_j=y[0+j:n+j]
            
            Ej=np.mean(y_sq*y_j)
            phi_j.append(Ej)
    else:
              
        for j in range(0,n):
            x_sq=x[0:N-j]**2  
            y_j=y[0+j:N]
            Ej=np.mean(x_sq*y_j)
            phi_j.append(Ej)
    phi_j=np.array(phi_j)
    
    test_hyp=0
    alpha=0.05
    Vx=np.var(phi_j,ddof=1)
    nor=ss.norm.ppf(1-alpha/2)*math.sqrt(Vx/n)
    interval=nor*math.sqrt(Vx/n)
    
    '''
    rss_s=np.sum((phi_j-test_hyp)**2)
    var_e=rss_s/(n-1)
    Vx=np.var(phi_j,ddof=1)
    se_1=math.sqrt(var_e/n/Vx)
    alpha=0.05
    nor=ss.norm.ppf(1-alpha/2)
    interval=nor*se_1
    #interval=nor*Vx**(1/2)
    '''
    return [phi_j,interval]


def pdf(X):
    peaks, _ =find_peaks(np.abs(X))
    peak_values=X[peaks]
    b_w=2*iqr(X)/len(X)**(1/3)
    n_bins=int(1.2*(max(X)-min(X))/b_w)
    hist = np.histogram(peak_values, bins=n_bins)
    hist_dist = scipy.stats.rv_histogram(hist)
    amp = np.linspace(min(X)*1.2, max(X)*1.2, n_bins)
    pdf=hist_dist.pdf(amp)
    Std_dv=np.std(X)
    #The skewness is a measure of how asymmetric a PDF is
    #Perfectly symmetric, skw=0
    Skw=skew(X)
    #The kurtosis is a measure of how ‘peaky’ the PDF is.
    #Flat, k=0
    Kurtos=kurtosis(X)
    return [pdf,amp,Std_dv,Skw,Kurtos,n_bins]
    
def Morlet(f,dt,n,a,b):
    #Morlet 
    n=np.arange(0,n)
    w=np.pi*2*f
    to = n*dt
    t = (to-b)/a
    psi=np.exp(1j*w*t)*np.exp(-t**2/2)
    psi_r=np.real(psi)
    psi_i=np.imag(psi)
    return [psi,psi_r,psi_i,to]


def morlet_wavelet(sig, fs, fmax, amin=1, amax=500):

    N=len(sig)
    t,dt = np.linspace(0, int(N/fs), N, retstep=True)
    w=(fmax/fs)*2*np.pi
    widths= np.arange(amin, amax)
    
    cwtm = signal.cwt(sig, signal.morlet2, widths,w=w)
    freq=w*fs/(2*np.pi*widths)
    a=fmax/freq
    return [t, a, np.abs(cwtm),freq]

    
def linear_fitting(X,y,intercept=True):
    X=np.array(X)
    y=np.array(y)
    if len(np.shape(X))==1:
        X=np.transpose(np.expand_dims(X, axis=0))
    if intercept:                                                 #Bias terms (column of ones)
        X=np.concatenate((X, np.ones((len(X),1))), axis=1)
    return np.matmul(np.linalg.pinv(X),y)                         #Return multiplication of pseudo-inverse of X times y.
    
def find_nearest(array,value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
        return array[idx-1]
    else:
        return array[idx]
