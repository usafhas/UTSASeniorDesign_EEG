# -*- coding: utf-8 -*-
"""
Created on Wed Dec 28 19:36:02 2016

@author: Heath
C:\Users\Heath\Anaconda2\python.exe -m pip install mne
"""
import mne
import numpy as np
import matplotlib.pyplot as plt
import pandas

""" Globals """
Window = 30 # Size of window to extract
Fs = 128
Sz = Window * Fs

norm = 4 # marker numbers in matrix
calm = 3
stress = 2

""" -------------------------------"""

def EDF_import(fname):
    # Read EDF File
    data = mne.io.read_raw_edf(fname, preload = True)
#    mne.viz.plot_raw(data, scalings='auto')
    return data
    
def EDF_event(data):
    data1 = mne.find_events(data)
#    mne.viz.plot_events(data1)
    return data1
    
def EDF_matrix(data):
    df = data.to_data_frame()
    df = df.transpose()
    M = df.as_matrix()
    return M
    
hRaw = './Data/Experiments/21Nov_Heath.edf'
mRaw = './Data/Experiments/21Nov_Matt.edf'

hData = EDF_import(hRaw)
mData = EDF_import(mRaw)

hEvent = EDF_event(hData)
hEvent[30,2] = 2 # Fix labeling error
mEvent = EDF_event(mData)

hMatrix = EDF_matrix(hData)
hMatrix[8,hEvent[30,0]] = 2
mMatrix = EDF_matrix(mData)

hBase = np.mean(hMatrix[0:8,0:Fs*5],axis=1)
mBase = np.mean(mMatrix[0:8,0:Fs*5],axis=1)

"""  Remove 5s base line from signals """
for i in range(0,8):
    hMatrix[i,:] = hMatrix[i,:]-hBase[i]
    mMatrix[i,:] = mMatrix[i,:]-mBase[i]


"""  Remove Epoch from raw data """
n = 0
s = 0
c = 0
step = 5 # size of moving window
slide = (60-Window)/step # each video is 60 seconds

sliden = (30-Window)/step # Because normal lenght is only 30 seconds

h={} # Declare a dictionary to store variables

x,y = np.shape(hMatrix)

for j in range(0,y):
    if hMatrix[8,j]==norm:
        if sliden == 0:
            h["norm{0}".format(n)]=hMatrix[0:8,j:j+Sz]
            n+=1
        else:
            for q in range(0,sliden):
                h["norm{0}".format(n)]=hMatrix[0:8,j+(q*step):j+Sz+(q*step)]
                n+=1
    elif hMatrix[8,j]==stress:
        for r in range(0,slide):
            h["stress{0}".format(s)]=hMatrix[0:8,j+(r*step):j+Sz+(r*step)]
            s+=1
    elif hMatrix[8,j]==calm:
        for r in range(0,slide):
            h["calm{0}".format(c)]=hMatrix[0:8,j+(r*step):j+Sz+(r*step)]
            c+=1
    else:
        pass
    
for j in range(0,y):
    if mMatrix[8,j]==norm:
        if sliden == 0:
            h["norm{0}".format(n)]=hMatrix[0:8,j:j+Sz]
            n+=1
        else:
            for q in range(0,sliden):
                h["norm{0}".format(n)]=mMatrix[0:8,j+(q*step):j+Sz+(q*step)]
                n+=1
    elif mMatrix[8,j]==stress:
        for r in range(0,slide):
            h["stress{0}".format(s)]=mMatrix[0:8,j+(r*step):j+Sz+(r*step)]
            s+=1
    elif mMatrix[8,j]==calm:
        for r in range(0,slide):
            h["calm{0}".format(c)]=mMatrix[0:8,j+(r*step):j+Sz+(r*step)]
            c+=1
    else:
        pass

    
    """ Save the Data into Numpy Arrays """
calm_list = []
for cc in range(0,c):        
    np.save('./Data/Experiments/calm{1}_W{0}'.format(Window,cc), h["calm{0}".format(cc)])
    calm_list.append('./Data/Experiments/calm{1}_W{0}'.format(Window,cc))
np.save('./Data/Experiments/calmW{0}'.format(Window), calm_list)

stress_list= []
for ss in range(0,s):
    np.save('./Data/Experiments/stress{1}_W{0}'.format(Window,ss), h["stress{0}".format(ss)])
    stress_list.append('./Data/Experiments/stress{1}_W{0}'.format(Window,ss))
np.save('./Data/Experiments/stressW{0}'.format(Window), stress_list)   

norm_list= []
for nn in range(0,n):
    np.save('./Data/Experiments/norm{1}_W{0}'.format(Window,nn), h["norm{0}".format(nn)])
    norm_list.append('./Data/Experiments/norm{1}_W{0}'.format(Window,nn))
np.save('./Data/Experiments/normW{0}'.format(Window), norm_list) 