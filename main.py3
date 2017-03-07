# -*- coding: utf-8 -*-
"""
Created on Fri Oct 07 14:32:23 2016

@author: Heath

Brain Rhythm 8 EEG Data

Full Buffer = [9,128 or "Sz"]
_________________________________________________________________________________________________
'Fp1'_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|
'Fp2'_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|
'Fz' _|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|
'C3' _|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|
'C4' _|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|
'Pz' _|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|
'O1' _|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|
'O2' _|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|
'STI014' _|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|

This matrix is all of the 'y' values, the amplitude of the signals
the 'x' is time, and is calculated by taking a range from 0 to Sz and dividing each by the sampling frequency


C:\\Users\Heath\Anaconda2\python.exe -m pip install

C:\\Users\Heath\Anaconda2\envs\SeniorDesign\python.exe -m pip install

pylsl

scikit-learn
scikit-fuzzy

C:\\Users\Heath\Google Drive\\UTSA\05_Fall 16\Senior Design\EEG_Project_Code\CODENAME_Duthess\PeakUtils-1.0.3.tar\dist\PeakUtils-1.0.3\PeakUtils-1.0.3\
python setup.py install

C:\\Users\Heath\Anaconda2\pyeeg-master\pyeeg-master python setup.py install
"""
from scipy import signal 
import LSL_importchunk  as lsl# import file with functions to grab data via LSL
# import EDF_MNE # file to import eeg data in edf
import Feature_calc2 as Feature_calc #calculate features from LSL
#import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.externals import joblib
import numpy as np
from pylsl import StreamInfo, StreamOutlet
import pickle
#import Thread_test
from time import sleep
from queue import Queue
from threading import Thread
#import sys
#import os

#%%
""" ================================= begin setups ================================="""
window = 15
clf = joblib.load('./SvCvN_W15_theta_sum.npy_QDA.pkl')
print("classifier loaded")
print("Imports Complete")



#%%      
""" ----------  Begin Main Loop of the program -----------------------------"""
if __name__=="__main__": # Main loop ------------------------------------------------------------------------
    """ Initialize the LSL stream inlet in LSL_importchunk.py  """
    print("looking for LSL.........Start the LSL")
    
    # initialize LSL grab
    inlet, buff = lsl.initialize_LSL()
    print("LSL initialized")
    #plt.close("all") # Close all open plots
    #setup LSL output stream
    # Name = UTSA, Content=Output, Channels=1, Hz=1, type = int, UID = SD
    info = StreamInfo('UTSA', 'Output', 1, 1, 'float32', 'SD')
    outlet = StreamOutlet(info)
    
    #----------------Global Variables----------------------------------
    """ Initialize global variable utilized throughout the code
    Set Fs to sampling rate of the headset
    Sz is size of our buffer, how many seconds of data do we want to store
    x is number of channels, BR8 transmits 8 + 1 for event/stimulation markers
    t = time of signal, it is 0 to size of signal divided by sampling rate to give seconds at each point"""
    
    Fs = 128.0 #sampling rate of the headset
    Sz = window*Fs # 1 second of sample
    x = 9 # number of channels + 1

    t = np.arange(0, Sz)*1/Fs
    print("Globals Declared")
    
    iii=0

    # get data for baseline removeal
    print("Collecting window for baseline")
    BSz = 3*Fs # 3 second baseline
    baselinebuff,x,y = lsl.Get_lsl(inlet,buff,BSz)
    base = np.mean(baselinebuff,axis=1)
    del buff  # Delete buffer to save memory
    buff = lsl.re(inlet)  # Reinitialize buff
    print("Baseline collected")
    
    Q = Queue(maxsize=20)

    while(True):
        inlet, buff = lsl.initialize_LSL()
        t = Thread(target=lsl.Buffer_thread, args=(inlet,buff,Sz,Q))
        t.daemon = True
        t.start()
#        del buff
        sleep(1)
        while not Q.empty():
    #        pass
        
    #    while Buffer_thread.hold(): # infinite Loop
            """ This is the main part of the code, and will run infinitely.  The buffer is filled and stored as Live_Matrix
        Live_matrix is then High passed, Normalized, and PSD feature calculated and returned
        finally the buffer is deleted, and then reinitialized to be sent back in to be filled 
            """
            print("Filling Buffer please wait, Buffer size = %d s"%window)
            fullbuff = Q.get()  # Get 9x128 Matrix from LSL
            fullbuff = np.nan_to_num(fullbuff)
            for i in range(0,x):
                fullbuff[i,:] = (fullbuff[i,:]-base[i])
                
            """ Open and Write JSON object """
            
            f_buffer = open('./buffer.json', 'w')
            f_psd = open('./psd.json', 'w')
            
            fullsum = np.sum(fullbuff, axis=0)  # collapse buffer channels to 1

            f_buffer.write('{\n\"buffer\":[')
            for n in fullsum:
                f_buffer.write(str(n))
                f_buffer.write(',')
            f_buffer.write(']\n}')
            
    #        np.save('./Data/Training/Raw/BR8/buffer_W{0}'.format(window),fullbuff)
            print("Buffer filled Preprocessing")
            live_M = Feature_calc.DEAP_process(fullbuff,Fs)
            Normalized, psdf, psdx = Feature_calc.process_live(live_M,Fs)
            alpha, beta, delta, gamma, theta = Feature_calc.Band_PSD(Normalized,Fs)
            
            
            f_psd.write('{\n\"psd\":[')
            for a in psdx:
                for n in a:
                    f_psd.write(str(n))
                    f_psd.write(',')
            f_psd.write(']\n}')
        
            f_buffer.close()
            f_psd.close()
            
            """ Select Feature to calculate =================================================="""
    
            feat = np.sum(theta,axis=0)
            """ ================================== Predict =============================="""
            
            feat.reshape(-1,1)
            result = clf.predict(feat)
            outlet.push_sample(result)
            
            print('I can see into the Future, I predict this to be ', result)
            print("================================", iii, "==============================")
            iii+=1
    ## -------------------------------------------------------------------------------

        