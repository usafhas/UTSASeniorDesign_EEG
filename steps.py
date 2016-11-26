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
from sklearn.decomposition import TruncatedSVD

#%%
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
#%%
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
    b = signal.firwin(65,[4.0],pass_zero=False,window='hamming',nyq=64.0)
	# b = firws(order, cutoff/nyq, windowarray)
    Mfilt = signal.lfilter(b,1.0,M,axis=1)

    # remove Baseline of signal
    Mfilt = signal.detrend(Mfilt, axis=1)
    
    # Bandstop filter
    bb = signal.firwin(65,[50.0, 60.0],window='hamming',nyq=64.0)
    
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
    M_normalized = M_normalized[[0,1,2,3,5,6,7],:]  # remove C4 channel
    
    return M_normalized
#%%	
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
#%%
plt.close('all')
Fs = 128.0	
WS = 15 # window size variable
t= np.arange(0,WS*Fs)*1/Fs
s1_happy = np.load('./Data/Training/happyW{0}.npy'.format(WS))
s1_angry = np.load('./Data/Training/angryW{0}.npy'.format(WS))
s1_sad = np.load('./Data/Training/normW{0}.npy'.format(WS))

m1 = np.load(s1_happy[1]+'.npy')
plt.figure()
plt.plot(t,m1[1,:])
plt.title('DEAP Raw')
m3f, m3p = Mpsd(m1,Fs)
plt.figure()
plt.plot(m3f,m3p[1,:])
plt.title('DEAP Raw PSD')

m2 = preprocess(m1,Fs)
plt.figure()
plt.plot(t,m2[1,:])
plt.title('DEAP Process')
m3f2, m3p2 = Mpsd(m2,Fs)
plt.figure()
#plt.semilogy(m3f2,m3p2[1,:])
plt.plot(m3f2,m3p2[1,:])
plt.title('DEAP Process PSD')


#%%
m4 = np.load(s1_happy[2]+'.npy')
n4 = np.load(s1_angry[2]+'.npy')
n5 = np.load(s1_angry[3]+'.npy')
test = np.vstack((m4,n4))

pca = PCA(n_components=3)
Principal = pca.fit_transform(test)
plt.figure()
plt.scatter(Principal[:,0],Principal[:,1])
#http://scikit-learn.org/stable/auto_examples/decomposition/plot_incremental_pca.html#sphx-glr-auto-examples-decomposition-plot-incremental-pca-py
princ = pca.fit_transform(n5)
plt.scatter(princ[:,0],princ[:,1],color='r')

#http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html#sklearn.decomposition.TruncatedSVD
svd = TruncatedSVD()
svd2 = svd.fit_transform(test)
plt.figure()
svd2.T
plt.scatter(svd2[0,:],svd2[1,:])

#http://scikit-learn.org/stable/modules/classes.html#module-sklearn.decomposition
#%%
#http://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis.html#sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
clf = QuadraticDiscriminantAnalysis()
lh = np.ones(8)
la = np.zeros(8)
y = np.append(lh,la)
m5 = np.load(s1_happy[2]+'.npy')
clf.fit(test, y)

print(clf.predict(n5))
print(clf.predict(m5))

#%%
#http://scikit-learn.org/stable/modules/generated/sklearn.cross_decomposition.CCA.html#sklearn.cross_decomposition.CCA
#http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier

#%%

#http://scikit-learn.org/0.17/modules/generated/sklearn.naive_bayes.GaussianNB.html#sklearn.naive_bayes.GaussianNB
MGNB = naive_bayes.GaussianNB()
clf = MGNB.fit(test,y)
print(clf.predict(n5))
print(clf.predict(m5))


#%%  Raw Brain Rhythm 8 buffer data
raw15 = np.load('./Data/Training/Raw/BR8/buffer_W15.npy')*1e-8
raw10 = np.load('./Data/Training/Raw/BR8/buffer_W15.npy')

rd15 = DEAP_process(raw15,Fs)
plt.figure()
plt.subplot(3,1,1)
plt.plot(t,m1[1,:],color='b')
plt.title('DEAP Raw')
plt.subplot(3,1,2)
plt.plot(t,raw15[1,:],color='r')
plt.title('BR8 Raw')
plt.subplot(3,1,3)
plt.plot(t,rd15[1,:],color='g')
plt.title('BR8 DEAP Blind Source')

rp15 = preprocess(rd15,Fs)
plt.figure()
plt.subplot(2,1,1)
plt.plot(t,m2[1,:],color='b')
plt.title('DEAP Preprocess')
plt.subplot(2,1,2)
plt.plot(t,rp15[1,:],color='r')
plt.title('BR8 Preprocess')

raw15f, raw15p = Mpsd(raw15,Fs)
rd15f, rd15p = Mpsd(rd15,Fs)
rp15f, rp15p = Mpsd(rp15,Fs)

plt.figure()
plt.title('Power Spectrum')
plt.plot(m3f,m3p[1,:],color='b',label='DEAP Raw PSD')
plt.plot(m3f,raw15p[1,:],color='r',label='BR8 Raw')
plt.plot(m3f,rd15p[1,:],color='g',label='BR8 DEAP processed PSD')
plt.legend()

plt.figure()
plt.plot(m3f2,m3p2[1,:],color='b',label='DEAP Processed PSD')
plt.plot(m3f,rp15p[1,:],color='r',label='BR8 Processed PSD')
plt.legend()

