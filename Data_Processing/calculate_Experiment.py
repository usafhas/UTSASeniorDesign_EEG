# -*- coding: utf-8 -*-
"""
Created on Sat Oct 15 19:41:56 2016

@author: Heath
"""

import cPickle
import numpy as np
import Feature_calc2 as Feature_calc
#windows = [10,15,30] # cant do 3 due to fft welch
windows = [10,15,30]
for sel in range(0,np.size(windows)):
    WS = windows[sel]# window size variable
    
    ex_norm = np.load('./Data/Experiments/normW{0}.npy'.format(WS))  #This is stressed Data
    ex_stress = np.load('./Data/Experiments/stressW{0}.npy'.format(WS))

    ex_calm = np.load('./Data/Experiments/calmW{0}.npy'.format(WS)) #This is Calm Data
    
    
    h = {}
    x = np.size(ex_norm)
    
    for i in range(0,x):
        h["norm{0}".format(i)]=np.load(ex_norm[i]+'.npy')
    #    
    s = {}
    y = np.size(ex_calm)
    for j in range(0,y):
        s["calm{0}".format(j)]=np.load(ex_calm[j]+'.npy')
        
    a = {}
    z = np.size(ex_stress)
    for k in range(0,z):
        a["stress{0}".format(k)]=np.load(ex_stress[k]+'.npy')
    
    
    """ ============== Globals ======================================== """
    ch = [0,16,18,6,24,15,13,31]
    Fs = 128 # Sampling Rate
    #Sz = 3*128 # window Size
    """ --------------------------------------------------------------- """
    
    """ ---------------------  PSD ------------------------"""
    hpsd ={}
    for i in range(0,x):
         h["norm{0}".format(i)]=Feature_calc.preprocess(h["norm{0}".format(i)], Fs)
         hpsd["hpsdf{0}".format(i)], hpsd["hpsdx{0}".format(i)] = Feature_calc.Mpsd(h["norm{0}".format(i)], Fs)
         np.save(ex_norm[i]+'_psd', hpsd["hpsdx{0}".format(i)])
    spsd={}
    for j in range(0,y):
         s["calm{0}".format(j)]=Feature_calc.preprocess(s["calm{0}".format(j)], Fs)
         spsd["spsdf{0}".format(j)], spsd["spsdx{0}".format(j)] = Feature_calc.Mpsd(s["calm{0}".format(j)], Fs)
         np.save(ex_calm[j]+'_psd', spsd["spsdx{0}".format(j)])
         
    apsd={}
    for k in range(0,z):
        a["stress{0}".format(k)]=Feature_calc.preprocess(a["stress{0}".format(k)], Fs)
        apsd["apsdf{0}".format(k)], apsd["apsdx{0}".format(k)] = Feature_calc.Mpsd(a["stress{0}".format(k)], Fs)
        np.save(ex_stress[k]+'_psd', apsd["apsdx{0}".format(k)])
    
    for i in range(0,x):
         
         hpsd["hpsdx_sum{0}".format(i)] = np.sum(hpsd["hpsdx{0}".format(i)],axis=0)
         np.save(ex_norm[i]+'_psd_sum', hpsd["hpsdx_sum{0}".format(i)])
         
    
    for j in range(0,y):
    
         spsd["spsdx_sum{0}".format(j)] = np.sum(spsd["spsdx{0}".format(j)],axis=0)
         np.save(ex_calm[j]+'_psd_sum', spsd["spsdx_sum{0}".format(j)])
         
    for k in range(0,z):
    
        apsd["apsdx_sum{0}".format(k)] = np.sum(apsd["apsdx{0}".format(k)],axis=0)
        np.save(ex_stress[k]+'_psd_sum', apsd["apsdx_sum{0}".format(k)])
    
        
    for i in range(0,x):
         
         hpsd["hpsdx_flat{0}".format(i)] = hpsd["hpsdx{0}".format(i)].flatten()
         np.save(ex_norm[i]+'_psd_flat', hpsd["hpsdx_flat{0}".format(i)])
         
    
    for j in range(0,y):
    
         spsd["spsdx_flat{0}".format(j)] = spsd["spsdx{0}".format(j)].flatten()
         np.save(ex_calm[j]+'_psd_flat', spsd["spsdx_flat{0}".format(j)])
         
    for k in range(0,z):
    
        apsd["apsdx_flat{0}".format(k)] = apsd["apsdx{0}".format(k)].flatten()
        np.save(ex_stress[k]+'_psd_flat', apsd["apsdx_flat{0}".format(k)])  
        
        
    """---------------------------------- More PSD ----------------------------"""    
        
    for i in range(0,x):
         
         hpsd["hpsdx_mean{0}".format(i)] = np.mean(hpsd["hpsdx{0}".format(i)],axis=0)
         np.save(ex_norm[i]+'_psd_mean', hpsd["hpsdx_mean{0}".format(i)])
         
    
    for j in range(0,y):
    
         spsd["spsdx_mean{0}".format(j)] = np.mean(spsd["spsdx{0}".format(j)],axis=0)
         np.save(ex_calm[j]+'_psd_mean', spsd["spsdx_mean{0}".format(j)])
         
    for k in range(0,z):
    
        apsd["apsdx_mean{0}".format(k)] = np.mean(apsd["apsdx{0}".format(k)],axis=0)
        np.save(ex_stress[k]+'_psd_mean', apsd["apsdx_mean{0}".format(k)]) 
        
    """ ============================= BANDs of Matricies ======================== """
    hband = {}
    sband ={}
    aband = {}
    for i in range(0,x):
         hband["alpha{0}".format(i)], hband["beta{0}".format(i)], hband["delta{0}".format(i)], hband["gamma{0}".format(i)], hband["theta{0}".format(i)]=Feature_calc.Band_PSD(h["norm{0}".format(i)], Fs)
         np.save(ex_norm[i]+'_alpha', hband["alpha{0}".format(i)])
         np.save(ex_norm[i]+'_beta', hband["beta{0}".format(i)])
         np.save(ex_norm[i]+'_delta', hband["delta{0}".format(i)])
         np.save(ex_norm[i]+'_gamma', hband["gamma{0}".format(i)])
         np.save(ex_norm[i]+'_theta', hband["theta{0}".format(i)])
         
    for j in range(0,y):
         sband["alpha{0}".format(j)], sband["beta{0}".format(j)], sband["delta{0}".format(j)], sband["gamma{0}".format(j)], sband["theta{0}".format(j)]=Feature_calc.Band_PSD(s["calm{0}".format(j)], Fs)
         np.save(ex_calm[j]+'_alpha', sband["alpha{0}".format(j)])
         np.save(ex_calm[j]+'_beta', sband["beta{0}".format(j)])
         np.save(ex_calm[j]+'_delta', sband["delta{0}".format(j)])
         np.save(ex_calm[j]+'_gamma', sband["gamma{0}".format(j)])
         np.save(ex_calm[j]+'_theta', sband["theta{0}".format(j)])
    
    for k in range(0,z):
         aband["alpha{0}".format(k)], aband["beta{0}".format(k)], aband["delta{0}".format(k)], aband["gamma{0}".format(k)], aband["theta{0}".format(k)]=Feature_calc.Band_PSD(a["stress{0}".format(k)], Fs)
         np.save(ex_stress[k]+'_alpha', aband["alpha{0}".format(k)])
         np.save(ex_stress[k]+'_beta', aband["beta{0}".format(k)])
         np.save(ex_stress[k]+'_delta', aband["delta{0}".format(k)])
         np.save(ex_stress[k]+'_gamma', aband["gamma{0}".format(k)])
         np.save(ex_stress[k]+'_theta', aband["theta{0}".format(k)])
           
        
        
    for i in range(0,x):
         hband["alpha_sum{0}".format(i)] = np.sum(hband["alpha{0}".format(i)], axis=0)
#         hband["beta_sum{0}".format(i)] = np.sum(hband["beta{0}".format(i)], axis=0)
         hband["delta_sum{0}".format(i)] = np.sum(hband["delta{0}".format(i)], axis=0)
         hband["gamma_sum{0}".format(i)] = np.sum(sband["gamma{0}".format(j)], axis=0)
         hband["theta_sum{0}".format(i)] = np.sum(sband["theta{0}".format(j)], axis=0)
         
         hband["theta_gamma{0}".format(i)] = zip(hband["theta_sum{0}".format(i)],hband["gamma_sum{0}".format(i)])
         np.save(ex_norm[i]+'_theta_gamma',hband["theta_gamma{0}".format(i)])
         np.save(ex_norm[i]+'_alpha_sum', hband["alpha_sum{0}".format(i)])
#         np.save(ex_norm[i]+'_beta_sum', hband["beta_sum{0}".format(i)])
         np.save(ex_norm[i]+'_delta_sum', hband["delta_sum{0}".format(i)])
         np.save(ex_norm[i]+'_gamma_sum', hband["gamma_sum{0}".format(i)])
         np.save(ex_norm[i]+'_theta_sum', hband["theta_sum{0}".format(i)])
         
    for j in range(0,y):
         sband["alpha_sum{0}".format(j)] = np.sum(sband["alpha{0}".format(j)],axis=0) 
#         sband["beta_sum{0}".format(j)] = np.sum(sband["beta{0}".format(j)], axis=0)
         sband["delta_sum{0}".format(j)] = np.sum(sband["delta{0}".format(j)], axis=0)
         sband["gamma_sum{0}".format(j)] = np.sum(sband["gamma{0}".format(j)],axis=0) 
         sband["theta_sum{0}".format(j)] = np.sum(sband["theta{0}".format(j)],axis=0)
         
         sband["theta_gamma{0}".format(j)] = zip(sband["theta_sum{0}".format(j)],sband["gamma_sum{0}".format(j)])
         np.save(ex_calm[j]+'_theta_gamma',sband["theta_gamma{0}".format(j)])
         np.save(ex_calm[j]+'_alpha_sum', sband["alpha_sum{0}".format(j)])
#         np.save(ex_calm[j]+'_beta_sum', sband["beta_sum{0}".format(j)])
         np.save(ex_calm[j]+'_delta_sum', sband["delta_sum{0}".format(j)])
         np.save(ex_calm[j]+'_gamma_sum', sband["gamma_sum{0}".format(j)])
         np.save(ex_calm[j]+'_theta_sum', sband["theta_sum{0}".format(j)])
         
    for k in range(0,z):
         aband["alpha_sum{0}".format(k)] = np.sum(aband["alpha{0}".format(k)],axis=0)
#         aband["beta_sum{0}".format(k)] = np.sum(aband["beta{0}".format(k)],axis=0)
         aband["delta_sum{0}".format(k)] = np.sum(aband["delta{0}".format(k)],axis=0)
         aband["gamma_sum{0}".format(k)] = np.sum(aband["gamma{0}".format(k)],axis=0)
         aband["theta_sum{0}".format(k)] = np.sum(aband["theta{0}".format(k)],axis=0)
         
         aband["theta_gamma{0}".format(k)] = zip(aband["theta_sum{0}".format(k)],aband["gamma_sum{0}".format(k)])
         np.save(ex_stress[k]+'_theta_gamma', aband["theta_gamma{0}".format(k)])
         np.save(ex_stress[k]+'_alpha_sum', aband["alpha_sum{0}".format(k)])
#         np.save(ex_stress[k]+'_beta_sum', aband["beta_sum{0}".format(k)])
         np.save(ex_stress[k]+'_delta_sum', aband["delta_sum{0}".format(k)])
         np.save(ex_stress[k]+'_gamma_sum', aband["gamma_sum{0}".format(k)])
         np.save(ex_stress[k]+'_theta_sum', aband["theta_sum{0}".format(k)])
    
    """ ========================== Absolute Power ============================ """
    
    """ take the abs_psd of each band, alpha, beta, delta, gamma and theta, and aggregate them into a vector that
        can be used as a feature for the classifier. """
    #
    #S01t1_happy_abspwr = Feature_calc.abs_psd_feature(S01t1_happy, Fs)
    #np.save('./Data/Training/ABS_PSD/S01t1_happy_W3O_abspwr',S01t1_happy_abspwr)
    
    hpwr = {}
    for i in range(0,x):
         hpwr["happy{0}".format(i)]=Feature_calc.total_bandpower_perchannel(h["norm{0}".format(i)], Fs)
         np.save(ex_norm[i]+'_abspwr', hpwr["happy{0}".format(i)])
    spwr ={}
    for j in range(0,y):
         spwr["sad{0}".format(j)]=Feature_calc.total_bandpower_perchannel(s["calm{0}".format(j)], Fs)
         np.save(ex_calm[j]+'_abspwr', spwr["sad{0}".format(j)])
         
    apwr = {}     
    for k in range(0,z):
        apwr["angry{0}".format(k)] = Feature_calc.total_bandpower_perchannel(a["stress{0}".format(k)],Fs)
        np.save(ex_stress[k]+'_abspwr', apwr["angry{0}".format(k)])
         
    
    for i in range(0,x):
         hpwr["happy_sum{0}".format(i)]=np.sum(hpwr["happy{0}".format(i)],axis = 0)
         np.save(ex_norm[i]+'_abspwr_sum', hpwr["happy_sum{0}".format(i)])
    
    for j in range(0,y):
         spwr["sad_sum{0}".format(j)]=np.sum(spwr["sad{0}".format(j)],axis=0)
         np.save(ex_calm[j]+'_abspwr_sum', spwr["sad_sum{0}".format(j)])
    
    for k in range(0,z):
        apwr["angry_sum{0}".format(k)] = np.sum(apwr["angry{0}".format(k)],axis=0)
        np.save(ex_stress[k]+'_abspwr_sum', apwr["angry_sum{0}".format(k)])
        
    """ ========================== Alpha Absolute Power ============================ """
    
    """ take the abs_psd alpha band and aggregate into a vector that
        can be used as a feature for the classifier. """
    
    hpwra = {}
    for i in range(0,x):
         hpwra["happy{0}".format(i)]=Feature_calc.abs_psd_alpha(h["norm{0}".format(i)], Fs)
         np.save(ex_norm[i]+'_abspwr_alpha8', hpwra["happy{0}".format(i)])
    spwra ={}
    for j in range(0,y):
         spwra["sad{0}".format(j)]=Feature_calc.abs_psd_alpha(s["calm{0}".format(j)], Fs)
         np.save(ex_calm[j]+'_abspwr_alpha8', spwra["sad{0}".format(j)])
         
    apwra = {}     
    for k in range(0,z):
        apwra["angry{0}".format(k)] = Feature_calc.abs_psd_alpha(a["stress{0}".format(k)],Fs)
        np.save(ex_stress[k]+'_abspwr_alpha8', apwra["angry{0}".format(k)])
         
    
    """=========================== PSD + Power ================================"""
    
    for i in range(0,x):
         hpwr["happy_PP{0}".format(i)]=np.hstack((hpwr["happy_sum{0}".format(i)],hpsd["hpsdx_sum{0}".format(i)]))
         np.save(ex_norm[i]+'_PP', hpwr["happy_PP{0}".format(i)])
    
    for j in range(0,y):
         spwr["sad_PP{0}".format(j)]=np.hstack((spwr["sad_sum{0}".format(j)],spsd["spsdx_sum{0}".format(j)]))
         np.save(ex_calm[j]+'_PP', spwr["sad_PP{0}".format(j)])
    
    for k in range(0,z):
    
        apwr["angry_PP{0}".format(k)] = np.hstack((apwr["angry_sum{0}".format(k)],apsd["apsdx_sum{0}".format(k)]))
        np.save(ex_stress[k]+'_PP', apwr["angry_PP{0}".format(k)])

    """==================================== Center of Gravity and Frequency variability """
    
    hcog ={}
    for i in range(0,x):
         hcog["happy{0}".format(i)] = Feature_calc.CoG(hpsd["hpsdf{0}".format(i)], hpsd["hpsdx{0}".format(i)])
         np.save(ex_norm[i]+'_cog', hcog["happy{0}".format(i)])
    scog={}
    for j in range(0,y):
         scog["sad{0}".format(j)] = Feature_calc.CoG(spsd["spsdf{0}".format(j)], spsd["spsdx{0}".format(j)])
         np.save(ex_calm[j]+'_cog', scog["sad{0}".format(j)])
         
    acog={}
    for k in range(0,z):
        acog["angry{0}".format(k)] = Feature_calc.CoG(apsd["apsdf{0}".format(k)], apsd["apsdx{0}".format(k)])
        np.save(ex_stress[k]+'_cog', acog["angry{0}".format(k)])
    
    """ Spectrogram avearge over time=================="""
    
    for i in range(0,x):
         hpsd["hspec{0}".format(i)] = Feature_calc.spectrogram_hanning(h["norm{0}".format(i)], Fs)
         np.save(ex_norm[i]+'_spec', hpsd["hspec{0}".format(i)])
    for j in range(0,y):
         spsd["sspec{0}".format(j)] = Feature_calc.spectrogram_hanning(s["calm{0}".format(j)], Fs)
         np.save(ex_calm[j]+'_spec', spsd["sspec{0}".format(j)])
    for k in range(0,z):
        apsd["aspec{0}".format(k)] = Feature_calc.spectrogram_hanning(a["stress{0}".format(k)], Fs)
        np.save(ex_stress[k]+'_spec', apsd["aspec{0}".format(k)])
    
    """ ========================= Fractal Dimension =============================="""
    #import Feature_calc # calculate features from LSL
    hFD = {}
    aFD = {}
    sFD ={}
    arousaFD = []
    valaFD = []
    aroushFD = []
    valhFD = []
    aroussFD = []
    valsFD = []
    
    for i in range(0,x):
         hFD["happya{0}".format(i)], hFD["happyv{0}".format(i)]=Feature_calc.hfd_valarous(h["norm{0}".format(i)])
         aroushFD.append(hFD["happya{0}".format(i)])
         valhFD.append(hFD["happyv{0}".format(i)])
         hFD["happy{0}".format(i)]=Feature_calc.hfd_feat(h["norm{0}".format(i)])
         np.save(ex_norm[i]+'_HFD', hFD["happy{0}".format(i)])

    
    for j in range(0,y):
         sFD["sada{0}".format(j)], sFD["sadv{0}".format(j)]=Feature_calc.hfd_valarous(s["calm{0}".format(j)])
         aroussFD.append(sFD["sada{0}".format(j)])
         valsFD.append(sFD["sadv{0}".format(j)])
         sFD["sad{0}".format(j)]=Feature_calc.hfd_feat(s["calm{0}".format(j)])
         np.save(ex_calm[j]+'_HFD', sFD["sad{0}".format(j)])

    
    for k in range(0,z):
         aFD["angrya{0}".format(k)], aFD["angryv{0}".format(k)]=Feature_calc.hfd_valarous(a["stress{0}".format(k)])
         arousaFD.append(aFD["angrya{0}".format(k)])
         valaFD.append(aFD["angryv{0}".format(k)])
         aFD["angry{0}".format(k)]=Feature_calc.hfd_feat(a["stress{0}".format(k)])
         np.save(ex_stress[k]+'_HFD', aFD["angry{0}".format(k)])

         
#    valtest = np.append(valaFD,valhFD)
#    varoustest = np.append(arousaFD,aroushFD)
#    print 'angry min %f max %f' %(np.min(valaFD), np.max(valaFD))
#    print 'happy min %f max %f' %( np.min(valhFD), np.max(valhFD))
