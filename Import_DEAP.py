# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 10:06:36 2016

@author: Heath

http://www.eecs.qmul.ac.uk/mmv/datasets/deap/

http://www.eecs.qmul.ac.uk/mmv/datasets/deap/readme.html#prep
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

1	Fp1	      17	Fp2
2	AF3       18	AF4
3	F3        19	Fz
4	F7        20	F4
5	FC5       21	F8
6	FC1       22	FC6
7	C3        23	FC2
8	T7        24	Cz
9	CP5       25	C4
10	CP1       26	T8
11	P3       27	CP6
12	P7       28	CP2
13	PO3      29	P4
14	O1       30	P8
15	Oz       31	PO4
16	Pz       32	O2
Low arrousal, low valence = Sad
High arrousal, low valence = angry
high arrousal, high valence = happy
low arrousal, high valence = relaxed
"""

import cPickle
import numpy as np


""" ========================= Globals ============================================= """
""" These are the channels that we need to correspond to our EEG headset """
ch = [0,16,18,6,24,15,13,31]
Fs = 128 # Sampling Rate
windows = [10,15,30]
for sel in range(0,np.size(windows)):
    WS = windows[sel]# window size variable

    Sz = WS*128 # window Size
    offset = 10*Fs
    #offset = 0
    cons=Fs*1e-6
    
    
    x = cPickle.load(open('./Data/s01.dat', 'rb'))
    """ ------------------------ SUBJECT 01 ------------------------------------------------"""
    """
    1	37	21	34106731	9	7.95	16.95	8.37	7.86	2
    1	5	20	5362005	8.26	7.91	16.17	7.19	8.13	1
    1	22	30	22211440	2.08	2.99	5.07	3.22	7.33	2
    1	23	29	23070480	1.36	2.27	3.63	3	8.14	2
    """
    
    """ ------------------------Time points ----------------------------------------"""
    
    S1t1s =  0*cons + offset# subject 1 trial 1 point
    S1t37s = 0*cons + offset
    S1t5s =  0*cons + offset
    S1t22s = 0*cons + offset
    S1t23s = 0*cons + offset
    #happy   0 
    S1t2s =  0* cons + offset
    
    S1t4s =  0* cons + offset
    S1t5s =  0* cons + offset
    S1t7s =  0* cons + offset
    S1t20s = 0* cons + offset
    
    #sad
    S1t9s =  0*cons + offset
    S1t10s = 0*cons + offset
    S1t21s = 0*cons + offset
    S1t27s = 0*cons + offset
    S1t30s = 0*cons + offset
    S1t12s = 0 * cons + offset # this time is sad
    
    #Angry   0
    S1t6s =  0 * cons + offset
    S1t3s =  0* cons + offset  # this time stamp is angry
    S1t8s =  0 * cons + offset
    
    S1t13s = 0 * cons + offset
    S1t15s = 0 * cons + offset
    S1t16s = 0 * cons + offset
    S1t17s = 0 * cons + offset
    S1t18s = 0 * cons + offset
    S1t26s = 0 * cons + offset
    S1t33s = 0 * cons + offset
    S1t34s = 0 * cons + offset
    
    
    
    """ ------------------------------------------------------------------------------"""
    
    S01 = x.get('data')
    S01lable = x.get('labels')
    
    """40 x 40 x 8064	 = video/trial x channel x data"""
    """get participant 1 trial 1, 40 channels x 63 seconds"""
    
    S01_T1 = S01[0,ch,:]
    
    """labels	40 x 4	video/trial x label (valence, arousal, dominance, liking)"""
    S01L = S01lable[0,:]
    
    
    S01t1_happy = S01_T1[:,S1t1s:S1t1s+Sz]  # this is a 1 second array from time of first video Happy
    """ export the Raw Arrays to numpy objects """
    np.save('./Data/Training/Raw/101calm_W{0}'.format(WS),S01t1_happy)
    
    
    np.save('./Data/Training/Raw/137happy_W{0}'.format(WS), S01[20,ch,S1t37s:S1t37s+Sz])
    
    np.save('./Data/Training/Raw/105happy_W{0}'.format(WS),S01[19,ch,S1t5s:S1t5s+Sz])
    
    np.save('./Data/Training/Raw/122sad_W{0}'.format(WS),S01[29,ch,S1t22s:S1t22s+Sz])
    
    np.save('./Data/Training/Raw/123sad_W{0}'.format(WS),S01[29,ch,S1t23s:S1t23s+Sz])
    
    np.save('./Data/Training/Raw/102happy_W{0}'.format(WS),S01[17,ch,S1t2s:S1t2s+Sz])
    
    
    
    np.save('./Data/Training/Raw/104happy_W{0}'.format(WS),S01[23,ch,S1t4s:S1t4s+Sz])
    
    np.save('./Data/Training/Raw/107happy_W{0}'.format(WS),S01[30,ch,S1t7s:S1t7s+Sz])
    
    np.save('./Data/Training/Raw/120happy_W{0}'.format(WS),S01[27,ch,S1t20s:S1t20s+Sz])
    
    np.save('./Data/Training/Raw/109sad_W{0}'.format(WS),S01[12,ch,S1t9s:S1t9s+Sz])
    
    np.save('./Data/Training/Raw/110sad_W{0}'.format(WS),S01[32,ch,S1t10s:S1t10s+Sz])
    
    np.save('./Data/Training/Raw/121sad_W{0}'.format(WS),S01[10,ch,S1t21s:S1t21s+Sz])
    
    np.save('./Data/Training/Raw/127sad_W{0}'.format(WS),S01[11,ch,S1t27s:S1t27s+Sz])
    
    np.save('./Data/Training/Raw/130sad_W{0}'.format(WS),S01[27,ch,S1t30s:S1t30s+Sz])
    
    np.save('./Data/Training/Raw/112sad_W{0}'.format(WS),S01[9,ch,S1t12s:S1t12s+Sz])
    """ ------------------Angry ---------------------------"""
    np.save('./Data/Training/Raw/106angry_W{0}'.format(WS),S01[30,ch,S1t6s:S1t6s+Sz])
    
    np.save('./Data/Training/Raw/103angry_W{0}'.format(WS),S01[3,ch,S1t3s:S1t3s+Sz])
    
    
    np.save('./Data/Training/Raw/108angry_W{0}'.format(WS),S01[38,ch,S1t8s:S1t8s+Sz])
    
    
    
    
    
    np.save('./Data/Training/Raw/113angry_W{0}'.format(WS),S01[34,ch,S1t13s:S1t13s+Sz])
    
    
    np.save('./Data/Training/Raw/115angry_W{0}'.format(WS),S01[14,ch,S1t15s:S1t15s+Sz])
    
    
    np.save('./Data/Training/Raw/116angry_W{0}'.format(WS),S01[16,ch,S1t16s:S1t16s+Sz])
    
    
    np.save('./Data/Training/Raw/117angry_W{0}'.format(WS),S01[35,ch,S1t17s:S1t17s+Sz])
    
    
    np.save('./Data/Training/Raw/118angry_W{0}'.format(WS),S01[33,ch,S1t18s:S1t18s+Sz])
    
    
    np.save('./Data/Training/Raw/126angry_W{0}'.format(WS),S01[31,ch,S1t26s:S1t26s+Sz])
    
    
    np.save('./Data/Training/Raw/133angry_W{0}'.format(WS),S01[37,ch,S1t33s:S1t33s+Sz])
    
    
    np.save('./Data/Training/Raw/134angry_W{0}'.format(WS),S01[36,ch,S1t34s:S1t34s+Sz])
    
    #np.save('./Data/Training/Raw/134norm_W{0}'.format(WS),S01[33,ch,S1t34s-Sz:S1t34s])
    
    """ ---------------------------- SUBJECT 02 ------------------------------------"""
    """
    2	19	8	26149925	9	9	18	9	9	
    2	27	25	32765179	9	9	18	8.1	9	
    2	28	39	33568406	1	1	2	2.97	1	
    2	31	21	35891446	1	1	2	5.04	1	
    """
    S2t19s = 0*cons + offset
    S2t27s = 0*cons + offset
    S2t28s = 0*cons + offset
    S2t31s = 0*cons + offset
    """ happy """          
    
    S2t8s =  0* cons + offset
    S2t9s =  0* cons + offset
    S2t11s = 0*cons + offset
    S2t12s = 0*cons + offset
    S2t17s = 0*cons + offset
    S2t18s = 0*cons + offset
    S2t19s = 0*cons + offset
    S2t20s = 0*cons + offset
                           
    """sad"""             
    S2t3s =  0* cons + offset
    S2t25s = 0*cons + offset
    S2t13s = 0*cons + offset #SAD
    
    """ --- Angry ---"""
    S2t4s =  0* cons + offset #Angry
    S2t5s = 0 *cons + offset
    S2t6s = 0 *cons + offset
    S2t7s = 0 *cons + offset
    S2t10s = 0*cons + offset
    S2t14s = 0*cons + offset
    
    
    """ -----Neutral-------"""
    S2t1s =  0* cons + offset
    S2t16s = 0*cons + offset #Neutral
    
    
    
    
    y = cPickle.load(open('./Data/s02.dat', 'rb'))
    S02 = y.get('data')
    
    
    
    np.save('./Data/Training/Raw/219happy_W{0}'.format(WS),S02[7,ch,S2t19s:S2t19s+Sz])
    
    np.save('./Data/Training/Raw/227happy_W{0}'.format(WS),S02[24,ch,S2t27s:S2t27s+Sz])
    
    np.save('./Data/Training/Raw/228sad_W{0}'.format(WS),S02[38,ch,S2t28s:S2t28s+Sz])
    
    np.save('./Data/Training/Raw/231sad_W{0}'.format(WS),S02[20,ch,S2t31s:S2t31s+Sz])
    
    np.save('./Data/Training/Raw/201norm_W{0}'.format(WS),S02[26,ch,S2t1s:S2t1s+Sz])
    
    np.save('./Data/Training/Raw/204angry_W{0}'.format(WS),S02[30,ch,S2t4s:S2t4s+Sz]) #angry
    
    np.save('./Data/Training/Raw/208happy_W{0}'.format(WS),S02[32,ch,S2t8s:S2t8s+Sz])
    
    np.save('./Data/Training/Raw/209happy_W{0}'.format(WS),S02[23,ch,S2t9s:S2t9s+Sz])
    
    np.save('./Data/Training/Raw/211happy_W{0}'.format(WS),S02[19,ch,S2t11s:S2t11s+Sz])
    
    np.save('./Data/Training/Raw/212happy_W{0}'.format(WS),S02[11,ch,S2t12s:S2t12s+Sz])
    
    np.save('./Data/Training/Raw/213sad_W{0}'.format(WS),S02[27,ch,S2t13s:S2t13s+Sz])
    
    np.save('./Data/Training/Raw/216norm_W{0}'.format(WS),S02[05,ch,S2t16s:S2t16s+Sz])
    
    np.save('./Data/Training/Raw/217happy_W{0}'.format(WS),S02[25,ch,S2t17s:S2t17s+Sz])
    
    np.save('./Data/Training/Raw/218happy_W{0}'.format(WS),S02[17,ch,S2t18s:S2t18s+Sz])
    
    np.save('./Data/Training/Raw/219happy_W{0}'.format(WS),S02[7,ch,S2t19s:S2t19s+Sz])
    
    np.save('./Data/Training/Raw/220happy_W{0}'.format(WS),S02[1,ch,S2t20s:S2t20s+Sz])
    
    np.save('./Data/Training/Raw/203sad_W{0}'.format(WS),S02[34,ch,S2t3s:S2t3s+Sz])
    
    np.save('./Data/Training/Raw/225sad_W{0}'.format(WS),S02[22,ch,S2t25s:S2t25s+Sz])
    
    """ -------------- Angry ------------------------------"""
    
    np.save('./Data/Training/Raw/205angry_W{0}'.format(WS),S02[37,ch,S2t5s:S2t5s+Sz])
    
    np.save('./Data/Training/Raw/206angry_W{0}'.format(WS),S02[31,ch,S2t6s:S2t6s+Sz])
    
    np.save('./Data/Training/Raw/207angry_W{0}'.format(WS),S02[29,ch,S2t7s:S2t7s+Sz])
    
    np.save('./Data/Training/Raw/210angry_W{0}'.format(WS),S02[28,ch,S2t10s:S2t10s+Sz])
    
    np.save('./Data/Training/Raw/214angry_W{0}'.format(WS),S02[36,ch,S2t14s:S2t14s+Sz])
    
    
    #np.save('./Data/Training/Raw/205norm_W{0}'.format(WS),S02[04,ch,S2t5s-Sz:S2t5s])
    #np.save('./Data/Training/Raw/206norm_W{0}'.format(WS),S02[05,ch,S2t6s-Sz:S2t6s])
    #np.save('./Data/Training/Raw/207norm_W{0}'.format(WS),S02[06,ch,S2t7s-Sz:S2t7s])
    #np.save('./Data/Training/Raw/210norm_W{0}'.format(WS),S02[9,ch, S2t10s-Sz:S2t10s])
    #np.save('./Data/Training/Raw/214norm_W{0}'.format(WS),S02[13,ch,S2t14s-Sz:S2t14s])
    
    """ =================== Subject 04 ========================="""
    z = cPickle.load(open('./Data/s04.dat', 'rb'))
    S04 = z.get('data')
    
    """ ---------- Time points --------------------------"""
    """ Sad"""
    s4t20s = 0*cons  + offset
    s4t21s = 0*cons  + offset
    s4t22s = 0*cons  + offset
    s4t23s = 0*cons  + offset
    s4t24s = 0*cons  + offset
    s4t30s = 0*cons  + offset
    s4t31s = 0*cons  + offset
    s4t36s = 0*cons  + offset
    s4t37s = 0*cons  + offset
    s4t38s = 0*cons  + offset
    
    """ angry """
    S4t25s = 0*cons  + offset
    S4t27s = 0*cons  + offset
    S4t29s = 0*cons  + offset
    S4t40s = 0*cons  + offset
    
    S4t04s = 0*cons  + offset
    
    
    
    np.save('./Data/Training/Raw/420sad_W{0}'.format(WS),S04[20,ch,s4t20s:s4t20s+Sz])
    
    np.save('./Data/Training/Raw/421sad_W{0}'.format(WS),S04[33,ch,s4t21s:s4t21s+Sz])
    
    np.save('./Data/Training/Raw/422sad_W{0}'.format(WS),S04[37,ch,s4t22s:s4t22s+Sz])
    
    np.save('./Data/Training/Raw/423sad_W{0}'.format(WS),S04[36,ch,s4t23s:s4t23s+Sz])
    
    np.save('./Data/Training/Raw/424sad_W{0}'.format(WS),S04[35,ch,s4t24s:s4t24s+Sz])
    
    np.save('./Data/Training/Raw/430sad_W{0}'.format(WS),S04[34,ch,s4t30s:s4t30s+Sz])
    
    np.save('./Data/Training/Raw/431sad_W{0}'.format(WS),S04[30,ch,s4t31s:s4t31s+Sz])
    
    np.save('./Data/Training/Raw/436sad_W{0}'.format(WS),S04[32,ch,s4t36s:s4t36s+Sz])
    
    np.save('./Data/Training/Raw/437sad_W{0}'.format(WS),S04[15,ch,s4t37s:s4t37s+Sz])
    
    np.save('./Data/Training/Raw/438sad_W{0}'.format(WS),S04[21,ch,s4t38s:s4t38s+Sz])
    
    
    """ --------------Angry --------------------------------"""
    
    np.save('./Data/Training/Raw/425angry_W{0}'.format(WS),S04[23,ch,S4t25s:S4t25s+Sz])
    
    np.save('./Data/Training/Raw/427angry_W{0}'.format(WS),S04[28,ch,S4t27s:S4t27s+Sz])
    
    np.save('./Data/Training/Raw/429angry_W{0}'.format(WS),S04[27,ch,S4t29s:S4t29s+Sz])
    
    np.save('./Data/Training/Raw/440angry_W{0}'.format(WS),S04[31,ch,S4t40s:S4t40s+Sz])
    
    np.save('./Data/Training/Raw/404norm_W{0}'.format(WS),S04[11,ch,S4t04s:S4t04s+Sz])
    #                                                             
    #np.save('./Data/Training/Raw/427norm_W{0}'.format(WS),S04[26,ch,S4t27s-Sz:S4t27s])
    #                                                             
    #np.save('./Data/Training/Raw/429norm_W{0}'.format(WS),S04[28,ch,S4t29s-Sz:S4t29s])
    #                                                             
    #np.save('./Data/Training/Raw/440norm_W{0}'.format(WS),S04[39,ch,S4t40s-Sz:S4t40s])
    
    
    """-------------------Nuetral Data -----------------"""
    x = cPickle.load(open('./Data/s08.dat', 'rb'))
    S08 = x.get('data')
    np.save('./Data/Training/Raw/821norm_W{0}'.format(WS),S08[37,ch,0+offset:0+offset+Sz])
    np.save('./Data/Training/Raw/815norm_W{0}'.format(WS),S08[27,ch,0+offset:0+offset+Sz])
    np.save('./Data/Training/Raw/807norm_W{0}'.format(WS),S08[6,ch,0+offset:0+offset+Sz])
    np.save('./Data/Training/Raw/832norm_W{0}'.format(WS),S08[25,ch,0+offset:0+offset+Sz])
	
    """----Some calm data for participant 09 ----"""
    np.save('./Data/Training/Raw/811calm_W{0}'.format(WS),S08[19,ch,0+offset:0+offset+Sz])
    np.save('./Data/Training/Raw/813calm_W{0}'.format(WS),S08[16,ch,0+offset:0+offset+Sz])
    np.save('./Data/Training/Raw/826calm_W{0}'.format(WS),S08[17,ch,0+offset:0+offset+Sz])
    np.save('./Data/Training/Raw/830calm_W{0}'.format(WS),S08[36,ch,0+offset:0+offset+Sz])
    np.save('./Data/Training/Raw/833calm_W{0}'.format(WS),S08[38,ch,0+offset:0+offset+Sz])
    np.save('./Data/Training/Raw/838calm_W{0}'.format(WS),S08[13,ch,0+offset:0+offset+Sz])
    	
    x = cPickle.load(open('./Data/s09.dat', 'rb'))
    S09 = x.get('data')
    np.save('./Data/Training/Raw/903norm_W{0}'.format(WS),S09[6,ch,0+offset:0+offset+Sz])
    np.save('./Data/Training/Raw/909norm_W{0}'.format(WS),S09[17,ch,0+offset:0+offset+Sz])
    np.save('./Data/Training/Raw/913norm_W{0}'.format(WS),S09[39,ch,0+offset:0+offset+Sz])
    np.save('./Data/Training/Raw/916norm_W{0}'.format(WS),S09[19,ch,0+offset:0+offset+Sz])
    np.save('./Data/Training/Raw/921norm_W{0}'.format(WS),S09[26,ch,0+offset:0+offset+Sz])
    np.save('./Data/Training/Raw/924norm_W{0}'.format(WS),S09[4,ch,0+offset:0+offset+Sz])
    np.save('./Data/Training/Raw/929norm_W{0}'.format(WS),S09[22,ch,0+offset:0+offset+Sz])
    	
    	
    """----Some calm data for participant 09 ----"""
    np.save('./Data/Training/Raw/901calm_W{0}'.format(WS),S09[0,ch,0+offset:0+offset+Sz])
    np.save('./Data/Training/Raw/902calm_W{0}'.format(WS),S09[8,ch,0+offset:0+offset+Sz])
    np.save('./Data/Training/Raw/917calm_W{0}'.format(WS),S09[24,ch,0+offset:0+offset+Sz])
    	
	
    
    
    """===================Samples Added by Matt==========================="""
    x = cPickle.load(open('./Data/s10.dat', 'rb'))
    S10 = x.get('data')
    np.save('./Data/Training/Raw/1001happy_W{0}'.format(WS),S10[10,ch,0+offset:0+offset+Sz])
    np.save('./Data/Training/Raw/1004happy_W{0}'.format(WS),S10[31,ch,0+offset:0+offset+Sz])
    np.save('./Data/Training/Raw/1005angry_W{0}'.format(WS),S10[36,ch,0+offset:0+offset+Sz])
    np.save('./Data/Training/Raw/1007happy_W{0}'.format(WS),S10[6 ,ch,0+offset:0+offset+Sz])
    np.save('./Data/Training/Raw/1009happy_W{0}'.format(WS),S10[18,ch,0+offset:0+offset+Sz])
    np.save('./Data/Training/Raw/1011sad_W{0}'.format(WS)  ,S10[21,ch,0+offset:0+offset+Sz])
    np.save('./Data/Training/Raw/1012angry_W{0}'.format(WS),S10[35,ch,0+offset:0+offset+Sz])
    np.save('./Data/Training/Raw/1013angry_W{0}'.format(WS),S10[1 ,ch,0+offset:0+offset+Sz])
    np.save('./Data/Training/Raw/1014sad_W{0}'.format(WS)  ,S10[25,ch,0+offset:0+offset+Sz])
    np.save('./Data/Training/Raw/1015angry_W{0}'.format(WS),S10[38,ch,0+offset:0+offset+Sz])
    np.save('./Data/Training/Raw/1016angry_W{0}'.format(WS),S10[34,ch,0+offset:0+offset+Sz])
    
    np.save('./Data/Training/Raw/1020angry_W{0}'.format(WS),S10[32,ch,0+offset:0+offset+Sz])
    np.save('./Data/Training/Raw/1022angry_W{0}'.format(WS),S10[30,ch,0+offset:0+offset+Sz])
    np.save('./Data/Training/Raw/1033angry_W{0}'.format(WS),S10[33,ch,0+offset:0+offset+Sz])
    np.save('./Data/Training/Raw/1038angry_W{0}'.format(WS),S10[20,ch,0+offset:0+offset+Sz])
    np.save('./Data/Training/Raw/1039angry_W{0}'.format(WS),S10[29,ch,0+offset:0+offset+Sz])
    np.save('./Data/Training/Raw/1040angry_W{0}'.format(WS),S10[37,ch,0+offset:0+offset+Sz])
    
    np.save('./Data/Training/Raw/1018happy_W{0}'.format(WS),S10[8 ,ch,0+offset:0+offset+Sz])
    np.save('./Data/Training/Raw/1027happy_W{0}'.format(WS),S10[3 ,ch,0+offset:0+offset+Sz])
	
    np.save('./Data/Training/Raw/1002calm_W{0}'.format(WS),S10[23 ,ch,0+offset:0+offset+Sz])
    np.save('./Data/Training/Raw/1003calm_W{0}'.format(WS),S10[8 ,ch,0+offset:0+offset+Sz])
    np.save('./Data/Training/Raw/1006calm_W{0}'.format(WS),S10[13 ,ch,0+offset:0+offset+Sz])
    np.save('./Data/Training/Raw/1008calm_W{0}'.format(WS),S10[11 ,ch,0+offset:0+offset+Sz])
    np.save('./Data/Training/Raw/1010calm_W{0}'.format(WS),S10[12 ,ch,0+offset:0+offset+Sz])
    np.save('./Data/Training/Raw/1017calm_W{0}'.format(WS),S10[17 ,ch,0+offset:0+offset+Sz])
    np.save('./Data/Training/Raw/1025calm_W{0}'.format(WS),S10[14 ,ch,0+offset:0+offset+Sz])
    np.save('./Data/Training/Raw/1026calm_W{0}'.format(WS),S10[19 ,ch,0+offset:0+offset+Sz])
    np.save('./Data/Training/Raw/1030calm_W{0}'.format(WS),S10[3 ,ch,0+offset:0+offset+Sz])
    
    
    
    """ =============== Samples ======================================="""
                   
    Data_S1_happy = ['./Data/Training/Raw/137happy_W{0}'.format(WS),'./Data/Training/Raw/105happy_W{0}'.format(WS),'./Data/Training/Raw/102happy_W{0}'.format(WS),
                   './Data/Training/Raw/104happy_W{0}'.format(WS),
                   './Data/Training/Raw/105happy_W{0}'.format(WS),'./Data/Training/Raw/107happy_W{0}'.format(WS),
                   './Data/Training/Raw/120happy_W{0}'.format(WS),'./Data/Training/Raw/219happy_W{0}'.format(WS),'./Data/Training/Raw/227happy_W{0}'.format(WS),
    			   './Data/Training/Raw/208happy_W{0}'.format(WS),'./Data/Training/Raw/209happy_W{0}'.format(WS),
    			   './Data/Training/Raw/211happy_W{0}'.format(WS),'./Data/Training/Raw/212happy_W{0}'.format(WS),
    			   './Data/Training/Raw/217happy_W{0}'.format(WS),'./Data/Training/Raw/218happy_W{0}'.format(WS),
    			   './Data/Training/Raw/219happy_W{0}'.format(WS),'./Data/Training/Raw/220happy_W{0}'.format(WS), 
    			   
    			   './Data/Training/Raw/1001happy_W{0}'.format(WS),'./Data/Training/Raw/1004happy_W{0}'.format(WS),'./Data/Training/Raw/1007happy_W{0}'.format(WS),
    			   './Data/Training/Raw/1009happy_W{0}'.format(WS),'./Data/Training/Raw/1027happy_W{0}'.format(WS),
    			   './Data/Training/Raw/1018happy_W{0}'.format(WS)]
                   
    np.save('./Data/Training/happyW{0}'.format(WS), Data_S1_happy)
    
    Data_S1_sad = ['./Data/Training/Raw/122sad_W{0}'.format(WS),'./Data/Training/Raw/123sad_W{0}'.format(WS),'./Data/Training/Raw/110sad_W{0}'.format(WS),'./Data/Training/Raw/121sad_W{0}'.format(WS),
                   './Data/Training/Raw/127sad_W{0}'.format(WS),'./Data/Training/Raw/130sad_W{0}'.format(WS),'./Data/Training/Raw/228sad_W{0}'.format(WS),'./Data/Training/Raw/231sad_W{0}'.format(WS),
    			   './Data/Training/Raw/203sad_W{0}'.format(WS),'./Data/Training/Raw/225sad_W{0}'.format(WS),
    			   './Data/Training/Raw/420sad_W{0}'.format(WS),'./Data/Training/Raw/421sad_W{0}'.format(WS),
    			   './Data/Training/Raw/422sad_W{0}'.format(WS),'./Data/Training/Raw/423sad_W{0}'.format(WS),
    			   './Data/Training/Raw/424sad_W{0}'.format(WS),'./Data/Training/Raw/430sad_W{0}'.format(WS),
    			   './Data/Training/Raw/431sad_W{0}'.format(WS),'./Data/Training/Raw/436sad_W{0}'.format(WS),
    			   './Data/Training/Raw/437sad_W{0}'.format(WS),'./Data/Training/Raw/438sad_W{0}'.format(WS),
    			   './Data/Training/Raw/112sad_W{0}'.format(WS),'./Data/Training/Raw/213sad_W{0}'.format(WS),
    			   './Data/Training/Raw/1011sad_W{0}'.format(WS),'./Data/Training/Raw/1014sad_W{0}'.format(WS)]
                   
    np.save('./Data/Training/sadW{0}'.format(WS), Data_S1_sad)
    
    Data_S1_angry = ['./Data/Training/Raw/106angry_W{0}'.format(WS),'./Data/Training/Raw/108angry_W{0}'.format(WS),
    				'./Data/Training/Raw/113angry_W{0}'.format(WS),
    				'./Data/Training/Raw/115angry_W{0}'.format(WS),'./Data/Training/Raw/116angry_W{0}'.format(WS),
    				'./Data/Training/Raw/117angry_W{0}'.format(WS),'./Data/Training/Raw/118angry_W{0}'.format(WS),
    				'./Data/Training/Raw/126angry_W{0}'.format(WS),'./Data/Training/Raw/133angry_W{0}'.format(WS),
    				'./Data/Training/Raw/134angry_W{0}'.format(WS),'./Data/Training/Raw/205angry_W{0}'.format(WS),
    				'./Data/Training/Raw/206angry_W{0}'.format(WS),'./Data/Training/Raw/207angry_W{0}'.format(WS),
    				'./Data/Training/Raw/210angry_W{0}'.format(WS),'./Data/Training/Raw/214angry_W{0}'.format(WS),
    				'./Data/Training/Raw/425angry_W{0}'.format(WS),'./Data/Training/Raw/427angry_W{0}'.format(WS),
    				'./Data/Training/Raw/429angry_W{0}'.format(WS),'./Data/Training/Raw/440angry_W{0}'.format(WS),
    				'./Data/Training/Raw/103angry_W{0}'.format(WS),'./Data/Training/Raw/204angry_W{0}'.format(WS),
    				'./Data/Training/Raw/1005angry_W{0}'.format(WS),'./Data/Training/Raw/1012angry_W{0}'.format(WS),
    				'./Data/Training/Raw/1013angry_W{0}'.format(WS),'./Data/Training/Raw/1015angry_W{0}'.format(WS),
    				'./Data/Training/Raw/1016angry_W{0}'.format(WS),'./Data/Training/Raw/1020angry_W{0}'.format(WS),
    				'./Data/Training/Raw/1022angry_W{0}'.format(WS),'./Data/Training/Raw/1033angry_W{0}'.format(WS),
    				'./Data/Training/Raw/1038angry_W{0}'.format(WS),'./Data/Training/Raw/1039angry_W{0}'.format(WS),
    				'./Data/Training/Raw/1040angry_W{0}'.format(WS)]
    
    np.save('./Data/Training/angryW{0}'.format(WS), Data_S1_angry)
    
    
    Data_S1_norm=['./Data/Training/Raw/201norm_W{0}'.format(WS),'./Data/Training/Raw/216norm_W{0}'.format(WS),'./Data/Training/Raw/404norm_W{0}'.format(WS),
    			  './Data/Training/Raw/821norm_W{0}'.format(WS),'./Data/Training/Raw/815norm_W{0}'.format(WS),'./Data/Training/Raw/807norm_W{0}'.format(WS),
    			  './Data/Training/Raw/832norm_W{0}'.format(WS),'./Data/Training/Raw/903norm_W{0}'.format(WS),'./Data/Training/Raw/909norm_W{0}'.format(WS),
    			  './Data/Training/Raw/913norm_W{0}'.format(WS),'./Data/Training/Raw/916norm_W{0}'.format(WS),'./Data/Training/Raw/921norm_W{0}'.format(WS),
    			  './Data/Training/Raw/924norm_W{0}'.format(WS),'./Data/Training/Raw/929norm_W{0}'.format(WS)]
    
    np.save('./Data/Training/normW{0}'.format(WS), Data_S1_norm)
	
    Data_S1_calm=['./Data/Training/Raw/1002calm_W{0}'.format(WS),'./Data/Training/Raw/1003calm_W{0}'.format(WS),'./Data/Training/Raw/1006calm_W{0}'.format(WS),
					'./Data/Training/Raw/1008calm_W{0}'.format(WS),'./Data/Training/Raw/1010calm_W{0}'.format(WS),'./Data/Training/Raw/1017calm_W{0}'.format(WS),
					'./Data/Training/Raw/1025calm_W{0}'.format(WS),'./Data/Training/Raw/1026calm_W{0}'.format(WS),'./Data/Training/Raw/1030calm_W{0}'.format(WS),
					'./Data/Training/Raw/901calm_W{0}'.format(WS),'./Data/Training/Raw/902calm_W{0}'.format(WS),'./Data/Training/Raw/917calm_W{0}'.format(WS),
					'./Data/Training/Raw/811calm_W{0}'.format(WS),'./Data/Training/Raw/813calm_W{0}'.format(WS),'./Data/Training/Raw/826calm_W{0}'.format(WS),
					'./Data/Training/Raw/830calm_W{0}'.format(WS),'./Data/Training/Raw/833calm_W{0}'.format(WS),'./Data/Training/Raw/838calm_W{0}'.format(WS)]
	
    np.save('./Data/Training/calmW{0}'.format(WS), Data_S1_calm)