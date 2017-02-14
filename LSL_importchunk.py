""" Created by Heath Spidle 30 September 2016

This Script will be used to save a Labstreaminglayer into a matrix with a length of 1 second for processing


https://github.com/sccn/labstreaminglayer/tree/master/LSL/liblsl-Python/examples

This program imports LSL stream into a matrix Channel size = N x sampling frequency size = Fs to get 1 second of data, 
this in placed into an NxFsx10 3 dimensional matrix so mathematical operations can be applied before being sent to 
into the classifier
"""


from pylsl import StreamInlet, resolve_stream
import numpy as np
#import pandas as pd
# Global Variable


def initialize_LSL():
    """ Initialize the LSL stream
    buff is initialized as the first grab off of the LSL strea, this is n 8x1 array """
    # first resolve an EEG stream on the lab network
    print("looking for an EEG stream...")
    streams = resolve_stream('type', 'EEG')
    # create a new inlet to read from the stream
    inlet = StreamInlet(streams[0])

    buff, timestamp =inlet.pull_sample()
    return inlet, buff

Fs = 128 #sampling rate of the headset
i = 0
########################################################################################
def fill_chunk(inlet, buff):
    """ append the next sample from the LSL stream to buff """
    # get a new sample (you can also omit the timestamp part if you're not
    # interested in it)
    inlet = inlet
    
    sample, timestamp = inlet.pull_sample()
    if timestamp:
#        print(sample)
        sample1 = np.asarray(sample)
        sample1 = np.nan_to_num(sample1)
        #chunktest = np.append(chunktest, np.asarray(sample), axis=0)
        #chunktest = np.concatenate((chunktest, sample1), axis=1)
        buff = np.column_stack((buff, sample1))
        
    return buff
##################################################################################
def csize(buff):
    """ get the current size of the buffer """
    ## Break the signal into 1s or arbitrary sized chunks
    x,y = buff.shape
    return x,y
#################################################################################
#    test = np.empty((x,Fs-1,10))
#    if y == Fs:
def fillM(test,buff,x,i,Sz):
    """ this function was to fill a test array with the buffer and append along a third axis, this is not used any longer
    """
    
    test[0:x,0:Sz-1,i] = buff[0:x,0:Sz-1]
    del buff
    #chunktest, timestamp =inlet.pull_sample()
         
    return test
    
def re(inlet):
    """ Reinitialized the buffer after deleting.  This ensures it is the appropriate data type, and creates a new 
    8x1 array to be filled again """
    buff, timestamp =inlet.pull_sample()
    return buff
    
    
    # http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html
def to_panda(test,i):
    """ This creates a pandas dataframe from the input matrix with channel names """             
    index = ['Fp1', 'Fp2', 'Fz', 'C3', 'C4', 'Pz', 'O1', 'O2', 'STI014']
         
    df = pd.DataFrame(test[:,:,i], index)
         # DataFrame.as_matrix(columns=None)
    return df
#%% Used in Live Program
def Get_lsl(inlet, buff, Sz):
    """ This function fills the buffer to specified size Sz
    while buffer is less than the Sz it will append a new column from the LSL input stream
    when buffer is equal to the Sz it will return it to the infinite loop for processing """
    y=0
    while( y < Sz):
       buff = fill_chunk(inlet,buff)
       x,y = csize(np.asarray(buff))
#       t1 = np.arange(0, y)*1/Fs
#       live_plot(ax1, l11,l12,l13,l14,l15,l16,l17,l18,buff ,t1) # plot live signal, slows it down considerably
    return buff, x, y

#%%

def Buffer_thread(inlet, buff,Sz,Q):
    fullbuff, x, y = Get_lsl(inlet, buff, Sz)
    if not Q.full():
        Q.put(fullbuff)
    
    
    
#while True:
#    # get a new sample (you can also omit the timestamp part if you're not
#    # interested in it)
#    chunk, timestamps = inlet.pull_chunk()
#    if timestamps:
#        print(timestamps, chunk)
#        chunktest.append(chunk)
    


