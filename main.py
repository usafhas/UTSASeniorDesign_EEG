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


C:\Users\Heath\Anaconda2\python.exe -m pip install

C:\Users\Heath\Anaconda2\envs\SeniorDesign\python.exe -m pip install

pylsl
pyaudio
scikit-learn
scikit-fuzzy
pyrem
C:\Users\Heath\Google Drive\UTSA\05_Fall 16\Senior Design\EEG_Project_Code\CODENAME_Duthess\PeakUtils-1.0.3.tar\dist\PeakUtils-1.0.3\PeakUtils-1.0.3\
python setup.py install

C:\Users\Heath\Anaconda2\pyeeg-master\pyeeg-master python setup.py install


"""
from scipy import signal 
import LSL_importchunk  as lsl# import file with functions to grab data via LSL
# import EDF_MNE # file to import eeg data in edf
import Feature_calc2 as Feature_calc #calculate features from LSL
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.externals import joblib
import numpy as np
from GUI import App
#%%
""" ================================= begin setups ================================="""
window = 10

#features = ['_gamma_sum.npy','_theta_sum.npy','_HFD.npy']
#feature = 0

#if feature == 0:
#   window = 10 # always angry
#    clf = joblib.load('./Data/Training/Live/17Nov/HvAvN_W10_gamma_sum.npy_KNN7_clf_82.6086956522.pkl')
#elif feature == 1:
#    window = 10
#    clf = joblib.load('./Data/Training/Live/17Nov/HvAvN_W10_theta_sum.npy_KNN7_clf_82.6086956522.pkl')
#elif feature ==2:
#    window = 15 # happy
#    clf = joblib.load('./Data/Training/Live/17Nov/HvAvN_W15_HFD.npy_KNN3_clf_78.2608695652.pkl')
#elif feature ==3:
#    clf = joblib.load('./Data/Training/Live/17Nov/HvAvN_W15_psd.npy_KNN5_clf_55.625.pkl')
#    window = 15
#else:
#    print "No valid classifier selected"
 
joblib.load('./Data/Classifier/Stress_Calm/HvAvN_W10_theta_sum.npy_KNN5_clf_827.5.pkl')

print "classifier loaded"
print "Imports Complete"
#%%      
""" ----------  Begin Main Loop of the program -----------------------------"""
if __name__=="__main__": # Main loop ------------------------------------------------------------------------
    """ Initialize the LSL stream inlet in LSL_importchunk.py  """
    print "looking for LSL.........Start the LSL"
    # initialize LSL grab
    inlet, buff = lsl.initialize_LSL()
    print "LSL initialized"
    plt.close("all") # Close all open plots
    
    #----------------Global Variables----------------------------------
    """ Initialize global variable utilized throughout the code
    Set Fs to sampling rate of the headset
    Sz is size of our buffer, how many seconds of data do we want to store
    x is number of channels, BR8 transmits 8 + 1 for event/stimulation markers
    t = time of signal, it is 0 to size of signal divided by sampling rate to give seconds at each point"""
    
    Fs = 128.0 #sampling rate of the headset
    Sz = window*Fs # 1 second of sample
    x = 9 # number of channels + 1
    threads = []
    t = np.arange(0, Sz)*1/Fs
    print "Globals Declared"
    
    i=0
    
    # initialize GUI
    app = App()
    app.setup()
    
    # get data for baseline removeal
    print("Collecting window for baseline")
    BSz = 3*Fs # 3 second baseline
    baselinebuff,x,y = lsl.Get_lsl(inlet,buff,BSz)
    base = np.mean(baselinebuff,axis=1)
    del buff  # Delete buffer to save memory
    buff = lsl.re(inlet)  # Reinitialize buff
    print("Baseline collected")
    
    
    
    while(True): # infinite Loop
        """ This is the main part of the code, and will run infinitely.  The buffer is filled and stored as Live_Matrix
    Live_matrix is then High passed, Normalized, and PSD feature calculated and returned
    finally the buffer is deleted, and then reinitialized to be sent back in to be filled 
        """
        print "Filling Buffer please wait, Buffer size = %d s"%window
        fullbuff, x, y = lsl.Get_lsl(inlet, buff, Sz)  # Get 9x128 Matrix from LSL
        fullbuff = numpy.nan_to_num(fullbuff)
        for i in range(0,x):
            fullbuff[i,:] = (fullbuff[i,:]-base[i])*1e-8

#        np.save('./Data/Training/Raw/BR8/buffer_W{0}'.format(window),fullbuff)
        print "Buffer filled Preprocessing"
        live_M = Feature_calc.DEAP_process(fullbuff,Fs)
        Normalized, psdf, psdx = Feature_calc.process_live(live_M,Fs)
        alpha, beta, delta, gamma, theta = Feature_calc.Band_PSD(Normalized,Fs)
        
        """ Select Feature to calculate =================================================="""

        feat = np.sum(theta, axis=0)

        
        """ ================================== Predict =============================="""
        
        feat.reshape(-1,1)
        result = clf.predict(feat)
        
        print 'I can see into the Future, I predict this to be ', result
        print "================================", i, "=============================="
        i+=1
        
        app.update_label(result % 3, i)
        
        if result == 1:
#            Feature_calc.play_wav('Happy_1.wav',1024)
            print '\r\n\r\n'
            print "Subject Happy"
            print '\r\n'
        elif result == 2:
#            Feature_calc.play_wav('Angry_1.wav',1024)
            print '\r\n\r\n'
            print "Subject Angry"
            print '\r\n'
        else:
            print '\r\n\r\n'
            print "Brain Dead"
            print '\r\n'

## -------------------------------------------------------------------------------
        del buff  # Delete buffer to save memory
        buff = lsl.re(inlet)  # Reinitialize buff
    