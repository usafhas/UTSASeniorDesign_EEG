# -*- coding: utf-8 -*-
"""
Created on Sat Oct 01 11:51:05 2016

@author: Heath

Features:

PSD - Power Spectral Density
ASM - Spectral Power Assymetry
CSP - Common Spatial Pattern
HOC - Higher Order Crossing
HOS - Higher Order Spectra
ASP - Assymetric Spatial Pattern
RPCA - Robust Principle Component Analysis https://github.com/fivetentaylor/rpyca
TGA - Trimmed Grassmann Average https://github.com/glennq/tga

Differential Laterality (DLAT)  - differential band power asymmetry
'Fp1', 'Fp2', 'Fz', 'C3', 'C4', 'Pz', 'O1', 'O2', 'STI014'
(FP1-Fp2),(C3-C4),(O1-O2)
The absolute power of a band is the integral of all of the power values within its frequency range
R = absolute power right
L = absolute power left
[(R-L)/(R + L)]

Differential Caudality (DCAU)
(FP1-O1),(FP2-O2), (Fz-Pz)
http://docs.scipy.org/doc/scipy/reference/tutorial/integrate.html

min max normilazation

https://github.com/breuderink/psychic/tree/master/psychic

https://github.com/breuderink/eegtools/blob/master/eegtools/spatfilt.py CSP

http://docs.scipy.org/doc/scipy/reference/signal.html

sklearn.cross_decomposition.CCA
canonical cross correlation analysis
"""
#%%
from __future__ import division
from scipy import signal
from sklearn import preprocessing
from sklearn import naive_bayes
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import peakutils
from scipy.stats import threshold
import scipy
from sklearn.decomposition import KernelPCA
from sklearn.decomposition import PCA
from sklearn.decomposition import SparsePCA
from sklearn.decomposition import FastICA

#import skfuzzy as fuzz
#import pyeeg
#import pyrem as pr

#%%  Preprocessing Steps
def process_live(Live_matrix,Fs):
    print "Processing the Data, What else may I do for you"

    Live_matrix = preprocess(Live_matrix, Fs)
    psdf1, psdx1 = Mpsd(Live_matrix, Fs) # PSD of the raw matrix, PSD values and frequency bins
    return Live_matrix, psdf1, psdx1
    

def DEAP_process(Live_matrix,Fs):
    Live_matrix = Live_matrix[0:8,:]
    nyq = 0.5*Fs
    low = 4.0/nyq
    high = 45.0/nyq
    b = signal.firwin(61,[4.0, 45.0],pass_zero=False,window='hamming',nyq=64.0)
#    b,a = signal.butter(4, [low, high], btype='bandpass',analog = False)    # 9th order bandstop filtering 50-60 hz
    Mpre = signal.lfilter(b,1.0,Live_matrix, axis=1)
#    np.array([np_convolve(xi, b, mode='valid') for xi in x])

        
    ica = FastICA()
    S_ = ica.fit_transform(Mpre)  # Reconstruct signals
    A_ = ica.mixing_
    A_=A_.T
    
    # Remove baseline of the signals
    x,y = np.shape(A_)
    A = np.zeros((x,y))
    for i in range(0,x):
        base = np.mean(A_,axis=1)
        A[i,:] = A_[i,:]-base[i]
    
    return A

""" -------------------------- Preprocess the signals ------------------------------------"""
def preprocess(M,Fs): 
    """ Preprocessing of the signal 
    First: Perform a high pass filter to get rid of any dc or signals below 1 Hz
    apply a Bandstop filter for A/C noise between 50-60 Hz
    Normalize the matrix setting all values between -1 and 1 
    """
    # http://scikit-learn.org/stable/modules/preprocessing.html
    nyq = 0.5*Fs
    # use hoffman filter
	# https://sccn.ucsd.edu/svn/software/tags/EEGLAB7_0_0_2beta/functions/popfunc/pop_eegfilt.m
	# pop_eegfilt https://github.com/widmann/firfilt/blob/master/pop_eegfiltnew.m
	# Hamming window filt order +1
    b = signal.firwin(61,[4.0],pass_zero=False,window='hamming',nyq=64.0)
	# b = firws(order, cutoff/nyq, windowarray)
    Mfilt = signal.lfilter(b,1.0,M,axis=1)

    # remove Baseline of signal
    Mfilt = signal.detrend(Mfilt, axis=1)
    
    # Bandstop filter
    bb = signal.firwin(61,[50.0, 60.0],window='hamming',nyq=64.0)
    
#    low = 50.0/nyq
#    high = 64.0/nyq    
    Mpre = signal.lfilter(bb,1.0,Mfilt,axis=1)
   
    # normalized Matrix M
    #sklearn.preprocessing.maxabs_scale
#    mm = np.max(Mpre)
#    mi = np.min(Mpre)
    x,y = np.shape(Mpre)
    M_normalized=np.zeros((x,y))
    for k in range(0,x):
        
        mm = np.max(Mpre[k,:])
        mi = np.min(Mpre[k,:])
        M_normalized[k,:] = (Mpre[k,:]-mi)/(mm-mi) # normalize to zero and 1
        M_normalized[k,:] = (M_normalized[k,:]*2) -1 # change range from -1 to 1 range = 1--1 + min
		#http://stackoverflow.com/questions/10364575/normalization-in-variable-range-x-y-in-matlab
    if x == 9:
        np.delete(M_normalized, (8), axis = 0)
    #M_normalized[8,:] = 0
#    M_normalized = M_normalized[[0,1,2,3,5,6,7],:]  # remove C4 channel
    
    return M_normalized
#%%   Power Features

""" --------------------- Power Spectral Density (peridogram algorithm)-----------------------------"""
def Mpsd(M,Fs):
    psdf = []
    psdx = []
#    psdf, psdx = signal.periodogram(M, Fs, axis=1)
	# https://docs.scipy.org/doc/scipy-0.18.1/reference/generated/scipy.signal.welch.html#scipy.signal.welch
    # https://docs.scipy.org/doc/scipy-0.18.1/reference/generated/scipy.signal.hanning.html#scipy.signal.hanning
	# hanning(M[, sym])	Return a Hann window.
 
    win = signal.get_window('hanning',4*Fs)
    psdf, psdx = signal.welch(M,Fs,window = win,nperseg=4*Fs, noverlap = 2*Fs,axis=1)
    
    return psdf, psdx

""" --------------------------------------------Band Pass ---------------------------------------------"""
def alpha(M, Fs):  
    """This function computes the band pass filter of all 5 frequency components of brain waves """
    # bands delta(1-3), theta(4-7), alpha(8-13), beta(14-30), gamma(31-43)
    nyq = 0.5*Fs
    lowa = 8.0/nyq
    higha = 13.0/nyq
    
	ba = signal.firwin(61,[lowa, higha],pass_zero=False,window='hamming')    
    Malpha = signal.lfilter(ba,1.0,M, axis=1)
    
    lowb = 14.0/nyq
    highb = 30.0/nyq
    
	bb = signal.firwin(61,[lowb, highb],pass_zero=False,window='hamming')	
    Mbeta = signal.lfilter(bb,1.0,M, axis=1)
    
    lowd = 1.0/nyq
    highd = 3.0/nyq
    
	bd = signal.firwin(61,[lowd, highd],pass_zero=False,window='hamming')    
    Mdelta = signal.lfilter(bd,1.0,M, axis=1)
        
    lowg = 31.0/nyq
    highg = 43.0/nyq
    
	bg = signal.firwin(61,[lowg, highg],pass_zero=False,window='hamming') 	
    Mgamma = signal.lfilter(bg,1.0,M, axis=1)
    
    lowt = 4.0/nyq
    hight = 7.0/nyq
    
	bt = signal.firwin(61,[lowt, hight],pass_zero=False,window='hamming')     
    Mtheta = signal.lfilter(bt,1.0,M, axis=1)    
    
    return Malpha, Mbeta, Mdelta, Mgamma, Mtheta

def absolute_PSD(band):
    """ This function calculates the total power in a given band, integrating to find the area under the curve.
    band is the PSD array of the filterd alpha, beta, delta, gamma or theta signal. """
    #low freq - lower limit of integrtion
    #high freq - upper limit of integration
    #PSD of band
#    abs_psd = integrate.trapz(band,axis =1)
    abs_psd = np.sum(band,axis=1)
    power = np.sum(abs_psd)
    
    
    return abs_psd, power
    
def abs_psd_feature(Live_matrix,Fs):
    """ take the abs_psd of each band, alpha, beta, delta, gamma and theta, and aggregate them into a vector that
    can be used as a feature for the classifier. """
    
    Malph, Mbeta, Mdelta, Mgamma, Mtheta = alpha(Live_matrix,Fs)
    psdfa, alphaa = Mpsd(Malph, Fs)
    psdfb, beta = Mpsd(Mbeta, Fs)
    psdfd, delta = Mpsd(Mdelta, Fs)
    psdfg, gamma = Mpsd(Mgamma, Fs)
    psdft, theta = Mpsd(Mtheta, Fs)
    
    alphaabs, powera = absolute_PSD(alphaa)
    betaabs, powerb = absolute_PSD(beta)
    deltaabs, powerd = absolute_PSD(delta)
    gammaabs, powerg = absolute_PSD(gamma)
    thetaabs, powert = absolute_PSD(theta)
    x,y = np.shape(Live_matrix)
    feature=np.zeros((1,8))
    """Make a 9x5 array, each row is a channel PSD, each column is a frequency band """
    feature = alphaabs.T
    feature = np.append(feature, betaabs)
    feature = np.append(feature, deltaabs)
    feature = np.append(feature, gammaabs)
    feature = np.append(feature, thetaabs)
    feature= np.reshape(feature,(1,x*5))
#    feature = powera
#    feature = np.column_stack((feature, powerb))
#    feature = np.column_stack((feature, powerd))
#    feature = np.column_stack((feature, powerg))
#    feature = np.column_stack((feature, powert))

    return feature
    
def Band_PSD(Live_matrix,Fs):
    """ take the abs_psd of each band, alpha, beta, delta, gamma and theta, and aggregate them into a vector that
    can be used as a feature for the classifier. """
    
    Malph, Mbeta, Mdelta, Mgamma, Mtheta = alpha(Live_matrix,Fs)
    psdfa, alphaa = Mpsd(Malph, Fs)
    psdfb, beta = Mpsd(Mbeta, Fs)
    psdfd, delta = Mpsd(Mdelta, Fs)
    psdfg, gamma = Mpsd(Mgamma, Fs)
    psdft, theta = Mpsd(Mtheta, Fs)

    return alphaa, beta, delta, gamma, theta

#%% Non Power Features =====================================================================	
def hfd_valarous(M):
    # higuchi fractal dimension
    #'Fp1', 'Fp2', 'Fz', 'C3', 'C4', 'Pz', 'O1', 'O2', 'STI014'
    x,y = np.shape(M)
    k = 2**3
    if x == 8:
        
        arous = hfd(M[5,:],k) #Ideally FC6 use PZ or C4
        val = hfd(M[4,:],k)-hfd(M[3,:],k) #Ideally AF3-F4 we do C4-FP1
    else: # if C4 is removed from matrix due to faulty sensor
        arous = hfd(M[4,:],k) #Ideally FC6 use PZ or C4
        val = hfd(M[4,:],k)-hfd(M[3,:],k)
    
    #(val-mi)/(mm-mi) # normalize to zero and 1
    #(val*max-min) + min
    minval = -0.15
    maxval = 0.15
    minarous = 1.30
    maxarous = 2.0
    
    pval = (((val-minval)/(maxval-minval))*(8))+1
    parous = (((arous-minarous)/(maxarous-minarous))*(4))+5
    
#    return arous, val
    return parous,pval
    
def hfd_feat(M):
    x,y = np.shape(M)
    feat = np.empty((1,x))
    for i in range(0,x):
        feat[0,i] = hfd(M[i,:],7)# 7 got a good value
    
    return feat
    
def hfd_specfeat(M):
    feat = []
    f, t, S = signal.spectrogram(M)
    x,y,z = np.shape(S)
    for i in range(0,z):
        for k in range(0,x):
            ch = hfd(S[k,:,i],7)
            feat = np.append(feat,ch)

    return feat
    
def DLAT(M,Fs):
    
    """ Differential Laterality (DLAT)  - differential band power asymmetry
    'Fp1', 'Fp2', 'Fz', 'C3', 'C4', 'Pz', 'O1', 'O2', 'STI014'
    (FP1-Fp2),(C3-C4),(O1-O2)   """ 
    x,y = np.shape(M)
    dlat = np.zeros((3,y))
    dcau = np.zeros((3,y))
    
    dlat[0] = M[0,:]-M[1,:]
    dlat[1] = M[3,:] - M[4,:]
    dlat[2]= M[6,:]-M[7,:]
    
    """Differential Caudality (DCAU)
    (FP1-O1),(FP2-O2), (Fz-Pz)"""
    dcau[0] = M[0,:]-M[6,:]
    dcau[1] = M[1,:]-M[7,:]
    dcau[2] = M[2,:]-M[5,:]
    dlatf, dlatp = Mpsd(dlat,Fs)
    dcauf, dcaup = Mpsd(dcau,Fs)
    D = np.append(dlat,dcau, axis=1)
    E = np.append(dlatp,dcaup, axis=1)
    return E
#csd(x, y[, fs, window, nperseg, noverlap, ...])	Estimate the cross power spectral density, Pxy, using Welch’s method.
#coherence(x, y[, fs, window, nperseg, ...])	Estimate the magnitude squared coherence estimate, Cxy, of discrete-time signals X and Y using Welch’s method.
#%%  Function definitions copied from other libraries
import pyaudio
import wave
import sys
import os.path
import time

CHUNK_SIZE = 1024

def play_wav(wav_filename, chunk_size=CHUNK_SIZE):
    '''
    Play (on the attached system sound device) the WAV file
    named wav_filename.
    '''

    try:
        print 'Trying to play file ' + wav_filename
        wf = wave.open(wav_filename, 'rb')
    except IOError as ioe:
        sys.stderr.write('IOError on file ' + wav_filename + '\n' + \
        str(ioe) + '. Skipping.\n')
        return
    except EOFError as eofe:
        sys.stderr.write('EOFError on file ' + wav_filename + '\n' + \
        str(eofe) + '. Skipping.\n')
        return

    # Instantiate PyAudio.
    p = pyaudio.PyAudio()

    # Open stream.
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
        channels=wf.getnchannels(),
        rate=wf.getframerate(),
                    output=True)

    data = wf.readframes(chunk_size)
    while len(data) > 0:
        stream.write(data)
        data = wf.readframes(chunk_size)

    # Stop stream.
    stream.stop_stream()
    stream.close()

    # Close PyAudio.
    p.terminate()
    return
	
def hfd(a, k_max):

    r"""
    Compute Higuchi Fractal Dimension of a time series.
    Vectorised version of the eponymous [PYEEG]_ function.
    .. note::
        **Difference with PyEEG:**
        Results is different from [PYEEG]_ which appears to have implemented an erroneous formulae.
        [HIG88]_ defines the normalisation factor as:
        .. math::
            \frac{N-1}{[\frac{N-m}{k} ]\dot{} k}
        [PYEEG]_ implementation uses:
        .. math::
            \frac{N-1}{[\frac{N-m}{k}]}
        The latter does *not* give the expected fractal dimension of approximately `1.50` for brownian motion (see example bellow).
    :param a: a one dimensional floating-point array representing a time series.
    :type a: :class:`~numpy.ndarray` or :class:`~pyrem.time_series.Signal`
    :param k_max: the maximal value of k
    :type k_max: int
    :return: Higuchi's fractal dimension; a scalar
    :rtype: float
    Example from [HIG88]_. This should produce a result close to `1.50`:
    >>> import numpy as np
    >>> import pyrem as pr
    >>> i = np.arange(2 ** 15) +1001
    >>> z = np.random.normal(size=int(2 ** 15) + 1001)
    >>> y = np.array([np.sum(z[1:j]) for j in i])
    >>> pr.univariate.hfd(y,2**8)
    """

    L = []
    x = []
    N = a.size


    # TODO this could be used to pregenerate k and m idxs ... but memory pblem?
    # km_idxs = np.triu_indices(k_max - 1)
    # km_idxs = k_max - np.flipud(np.column_stack(km_idxs)) -1
    # km_idxs[:,1] -= 1
    #

    for k in xrange(1,k_max):
        Lk = 0
        for m in xrange(0,k):
            #we pregenerate all idxs
            idxs = np.arange(1,int(np.floor((N-m)/k)),dtype=np.int32)

            Lmk = np.sum(np.abs(a[m+idxs*k] - a[m+k*(idxs-1)]))
            Lmk = (Lmk*(N - 1)/(((N - m)/ k)* k)) / k
            Lk += Lmk


        L.append(np.log(Lk/(m+1)))
        x.append([np.log(1.0/ k), 1])

    (p, r1, r2, s)=np.linalg.lstsq(x, L)
    return p[0]
    
def stft(x, fs, framesz, hop):
    framesamp = int(framesz*fs) # with a frame size of 50 milliseconds
    hopsamp = int(hop*fs) # and hop size of 25 milliseconds.
    w = scipy.hanning(framesamp)
    X = scipy.array([scipy.fft(w*x[i:i+framesamp]) 
                     for i in range(0, len(x)-framesamp, hopsamp)])
    return X

def istft(X, fs, T, hop):
    x = scipy.zeros(T*fs)# lasting 5 seconds T
    framesamp = X.shape[1]
    hopsamp = int(hop*fs)
    for n,i in enumerate(range(0, len(x)-framesamp, hopsamp)):
        x[i:i+framesamp] += scipy.real(scipy.ifft(X[n]))
    return x
      	
#%% Unused Definitions / Unuseful =====================================================================================

def Mpsd2(M,Fs):
    psdf = []
    psdx = []
    psdf, psdx = signal.periodogram(M, Fs, axis=1)
    
    x,y = np.shape(psdx)
    
    for i in range (0,x):
        temp1 = psdf[:]*(i+1)
        temp2 = psdx[i,:]
        if i == 0:
                a = temp1
                b = temp2
        else:
            a = np.append(a,temp1)
            b = np.append(b,temp2)
    psd = zip(a,b)
#    psd = np.array(psd).T
#    for i in range (0,x):
#        a = zip(psdf[:]*(i+1),psdx[i,:])
#        if i == 0:
#            psd = a
##            psd = psd[:,:, np.newaxis]
#        else:
##            psd = np.dstack((psd,a))
#            psd = np.vstack((psd,a))
##    psd = np.swapaxes(psd,1,0)    

    return psd
	
def live_plot(ax, line1, line2, line3, line4, line5, line6,line7,line8, Live_matrix, t):
    """ This is the plotting function that will reside inside the infinite loop.  Using the lines obtained from
    init_liveplot(), this updates only the lines with the matrix data """
#    ax1.clear()
    line1.set_xdata(t[:])
    line2.set_xdata(t[:]) 
    line3.set_xdata(t[:]) 
    line4.set_xdata(t[:]) 
    line5.set_xdata(t[:]) 
    line6.set_xdata(t[:])
    line7.set_xdata(t[:])
    line8.set_xdata(t[:])
    line1.set_ydata(Live_matrix[0,:])
    line2.set_ydata(Live_matrix[1,:])
    line3.set_ydata(Live_matrix[2,:])
    line4.set_ydata(Live_matrix[3,:]) 
    line5.set_ydata(Live_matrix[4,:]) 
    line6.set_ydata(Live_matrix[5,:])
    line7.set_ydata(Live_matrix[6,:])
    line8.set_ydata(Live_matrix[7,:])
    
    plt.pause(0.000001)

    return   
    
def init_liveplot(lx,lyl, lyu, title):
    """ This function initiates the live plotting for the signals.  just using the plt.plot() command
    redraws the entire figure every time, this is very resouce intensive.  This method initializes the figures
    and the subplots and gets the line object from each subplot.  only the lines are updated on each plot
    command and runs much faster and smoother. """
    
#    fig = plt.figure(i)
    f, axarr = plt.subplots(4,2, sharex=True)
#    plt.ion()
    line = axarr
    plt.suptitle(title)
    
    for j in range(0,4):
        for i in range(0,2):
        
            axarr[j,i].set_xlim(0, lx)
            axarr[j,i].set_ylim(lyl, lyu)
    line1, = axarr[0,0].plot([],[])
    line2, = axarr[0,1].plot([],[]) 
    line3, = axarr[1,0].plot([],[]) 
    line4, = axarr[1,1].plot([],[]) 
    line5, = axarr[2,0].plot([],[]) 
    line6, = axarr[2,1].plot([],[]) 
    line7, = axarr[3,0].plot([],[]) 
    line8, = axarr[3,1].plot([],[]) 

    return axarr, line1, line2, line3, line4, line5, line6,line7,line8