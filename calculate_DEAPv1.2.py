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
    
    s1_happy = np.load('./Data/Training/happyW{0}.npy'.format(WS))
    s1_angry = np.load('./Data/Training/angryW{0}.npy'.format(WS))
    s1_sad = np.load('./Data/Training/normW{0}.npy'.format(WS))
    #s1_sad = np.load('./Data/Training/sadW{0}.npy'.format(WS))
    
    
    h = {}
    x = np.size(s1_happy)
    
    for i in range(0,x):
        h["happy{0}".format(i)]=np.load(s1_happy[i]+'.npy')
    #    
    s = {}
    y = np.size(s1_sad)
    for j in range(0,y):
        s["sad{0}".format(j)]=np.load(s1_sad[j]+'.npy')
        
    #s = {}
    #y = np.size(s1_norm)
    #for j in range(0,y):
    #    s["sad{0}".format(j)]=np.load(s1_norm[j]+'.npy')
        
    a = {}
    z = np.size(s1_angry)
    for k in range(0,z):
        a["angry{0}".format(k)]=np.load(s1_angry[k]+'.npy')
    
    
    """ ============== Globals ======================================== """
    ch = [0,16,18,6,24,15,13,31]
    Fs = 128 # Sampling Rate
    #Sz = 3*128 # window Size
    """ --------------------------------------------------------------- """
    
    """ --------------------- Power Spectral Density ------------------"""
    #S01t1_happy = Feature_calc.preprocess(S01t1_happy, Fs)
    #psdf, S01t1_happy_psdx = Feature_calc.Mpsd(S01t1_happy, Fs)
    #np.save('./Data/Training/S01t1_happy_W3O_PSD',S01t1_happy_psdx)
    #np.save('./Data/Training/psdf', psdf)
    
    """ ---------------------  PSD ------------------------"""
    hpsd ={}
    for i in range(0,x):
         h["happy{0}".format(i)]=Feature_calc.preprocess(h["happy{0}".format(i)], Fs)
         hpsd["hpsdf{0}".format(i)], hpsd["hpsdx{0}".format(i)] = Feature_calc.Mpsd(h["happy{0}".format(i)], Fs)
         np.save(s1_happy[i]+'_psd', hpsd["hpsdx{0}".format(i)])
    spsd={}
    for j in range(0,y):
         s["sad{0}".format(j)]=Feature_calc.preprocess(s["sad{0}".format(j)], Fs)
         spsd["spsdf{0}".format(j)], spsd["spsdx{0}".format(j)] = Feature_calc.Mpsd(s["sad{0}".format(j)], Fs)
         np.save(s1_sad[j]+'_psd', spsd["spsdx{0}".format(j)])
         
    apsd={}
    for k in range(0,z):
        a["angry{0}".format(k)]=Feature_calc.preprocess(a["angry{0}".format(k)], Fs)
        apsd["apsdf{0}".format(k)], apsd["apsdx{0}".format(k)] = Feature_calc.Mpsd(a["angry{0}".format(k)], Fs)
        np.save(s1_angry[k]+'_psd', apsd["apsdx{0}".format(k)])
    
    for i in range(0,x):
         
         hpsd["hpsdx_sum{0}".format(i)] = np.sum(hpsd["hpsdx{0}".format(i)],axis=0)
         np.save(s1_happy[i]+'_psd_sum', hpsd["hpsdx_sum{0}".format(i)])
         
    
    for j in range(0,y):
    
         spsd["spsdx_sum{0}".format(j)] = np.sum(spsd["spsdx{0}".format(j)],axis=0)
         np.save(s1_sad[j]+'_psd_sum', spsd["spsdx_sum{0}".format(j)])
         
    for k in range(0,z):
    
        apsd["apsdx_sum{0}".format(k)] = np.sum(apsd["apsdx{0}".format(k)],axis=0)
        np.save(s1_angry[k]+'_psd_sum', apsd["apsdx_sum{0}".format(k)])
    
        
    for i in range(0,x):
         
         hpsd["hpsdx_flat{0}".format(i)] = hpsd["hpsdx{0}".format(i)].flatten()
         np.save(s1_happy[i]+'_psd_flat', hpsd["hpsdx_flat{0}".format(i)])
         
    
    for j in range(0,y):
    
         spsd["spsdx_flat{0}".format(j)] = spsd["spsdx{0}".format(j)].flatten()
         np.save(s1_sad[j]+'_psd_flat', spsd["spsdx_flat{0}".format(j)])
         
    for k in range(0,z):
    
        apsd["apsdx_flat{0}".format(k)] = apsd["apsdx{0}".format(k)].flatten()
        np.save(s1_angry[k]+'_psd_flat', apsd["apsdx_flat{0}".format(k)])  
        
        
    """---------------------------------- More PSD ----------------------------"""    
        
    for i in range(0,x):
         
         hpsd["hpsdx_mean{0}".format(i)] = np.mean(hpsd["hpsdx{0}".format(i)],axis=0)
         np.save(s1_happy[i]+'_psd_mean', hpsd["hpsdx_mean{0}".format(i)])
         
    
    for j in range(0,y):
    
         spsd["spsdx_mean{0}".format(j)] = np.mean(spsd["spsdx{0}".format(j)],axis=0)
         np.save(s1_sad[j]+'_psd_mean', spsd["spsdx_mean{0}".format(j)])
         
    for k in range(0,z):
    
        apsd["apsdx_mean{0}".format(k)] = np.mean(apsd["apsdx{0}".format(k)],axis=0)
        np.save(s1_angry[k]+'_psd_mean', apsd["apsdx_mean{0}".format(k)]) 
        
    """ ============================= BANDs of Matricies ======================== """
    hband = {}
    sband ={}
    aband = {}
    for i in range(0,x):
         hband["alpha{0}".format(i)], hband["beta{0}".format(i)], hband["delta{0}".format(i)], hband["gamma{0}".format(i)], hband["theta{0}".format(i)]=Feature_calc.Band_PSD(h["happy{0}".format(i)], Fs)
         np.save(s1_happy[i]+'_alpha', hband["alpha{0}".format(i)])
         np.save(s1_happy[i]+'_beta', hband["beta{0}".format(i)])
         np.save(s1_happy[i]+'_delta', hband["delta{0}".format(i)])
         np.save(s1_happy[i]+'_gamma', hband["gamma{0}".format(i)])
         np.save(s1_happy[i]+'_theta', hband["theta{0}".format(i)])
         
    for j in range(0,y):
         sband["alpha{0}".format(j)], sband["beta{0}".format(j)], sband["delta{0}".format(j)], sband["gamma{0}".format(j)], sband["theta{0}".format(j)]=Feature_calc.Band_PSD(s["sad{0}".format(j)], Fs)
         np.save(s1_sad[j]+'_alpha', sband["alpha{0}".format(j)])
         np.save(s1_sad[j]+'_beta', sband["beta{0}".format(j)])
         np.save(s1_sad[j]+'_delta', sband["delta{0}".format(j)])
         np.save(s1_sad[j]+'_gamma', sband["gamma{0}".format(j)])
         np.save(s1_sad[j]+'_theta', sband["theta{0}".format(j)])
    
    for k in range(0,z):
         aband["alpha{0}".format(k)], aband["beta{0}".format(k)], aband["delta{0}".format(k)], aband["gamma{0}".format(k)], aband["theta{0}".format(k)]=Feature_calc.Band_PSD(a["angry{0}".format(k)], Fs)
         np.save(s1_angry[k]+'_alpha', aband["alpha{0}".format(k)])
         np.save(s1_angry[k]+'_beta', aband["beta{0}".format(k)])
         np.save(s1_angry[k]+'_delta', aband["delta{0}".format(k)])
         np.save(s1_angry[k]+'_gamma', aband["gamma{0}".format(k)])
         np.save(s1_angry[k]+'_theta', aband["theta{0}".format(k)])
           
        
        
    for i in range(0,x):
         hband["alpha_sum{0}".format(i)] = np.sum(hband["alpha{0}".format(i)], axis=0)
         hband["beta_sum{0}".format(i)] = np.sum(hband["beta{0}".format(i)], axis=0)
         hband["delta_sum{0}".format(i)] = np.sum(hband["delta{0}".format(i)], axis=0)
         hband["gamma_sum{0}".format(i)] = np.sum(sband["gamma{0}".format(j)], axis=0)
         hband["theta_sum{0}".format(i)] = np.sum(sband["theta{0}".format(j)], axis=0)
         
         hband["theta_gamma{0}".format(i)] = zip(hband["theta_sum{0}".format(i)],hband["gamma_sum{0}".format(i)])
         np.save(s1_happy[i]+'_theta_gamma',hband["theta_gamma{0}".format(i)])
         np.save(s1_happy[i]+'_alpha_sum', hband["alpha_sum{0}".format(i)])
         np.save(s1_happy[i]+'_beta_sum', hband["beta_sum{0}".format(i)])
         np.save(s1_happy[i]+'_delta_sum', hband["delta_sum{0}".format(i)])
         np.save(s1_happy[i]+'_gamma_sum', hband["gamma_sum{0}".format(i)])
         np.save(s1_happy[i]+'_theta_sum', hband["theta_sum{0}".format(i)])
         
    for j in range(0,y):
         sband["alpha_sum{0}".format(j)] = np.sum(sband["alpha{0}".format(j)],axis=0) 
         sband["beta_sum{0}".format(j)] = np.sum(sband["beta{0}".format(j)], axis=0)
         sband["delta_sum{0}".format(j)] = np.sum(sband["delta{0}".format(j)], axis=0)
         sband["gamma_sum{0}".format(j)] = np.sum(sband["gamma{0}".format(j)],axis=0) 
         sband["theta_sum{0}".format(j)] = np.sum(sband["theta{0}".format(j)],axis=0)
         
         sband["theta_gamma{0}".format(j)] = zip(sband["theta_sum{0}".format(j)],sband["gamma_sum{0}".format(j)])
         np.save(s1_sad[j]+'_theta_gamma',sband["theta_gamma{0}".format(j)])
         np.save(s1_sad[j]+'_alpha_sum', sband["alpha_sum{0}".format(j)])
         np.save(s1_sad[j]+'_beta_sum', sband["beta_sum{0}".format(j)])
         np.save(s1_sad[j]+'_delta_sum', sband["delta_sum{0}".format(j)])
         np.save(s1_sad[j]+'_gamma_sum', sband["gamma_sum{0}".format(j)])
         np.save(s1_sad[j]+'_theta_sum', sband["theta_sum{0}".format(j)])
         
    for k in range(0,z):
         aband["alpha_sum{0}".format(k)] = np.sum(aband["alpha{0}".format(k)],axis=0)
         aband["beta_sum{0}".format(k)] = np.sum(aband["beta{0}".format(k)],axis=0)
         aband["delta_sum{0}".format(k)] = np.sum(aband["delta{0}".format(k)],axis=0)
         aband["gamma_sum{0}".format(k)] = np.sum(aband["gamma{0}".format(k)],axis=0)
         aband["theta_sum{0}".format(k)] = np.sum(aband["theta{0}".format(k)],axis=0)
         
         aband["theta_gamma{0}".format(k)] = zip(aband["theta_sum{0}".format(k)],aband["gamma_sum{0}".format(k)])
         np.save(s1_angry[k]+'_theta_gamma', aband["theta_gamma{0}".format(k)])
         np.save(s1_angry[k]+'_alpha_sum', aband["alpha_sum{0}".format(k)])
         np.save(s1_angry[k]+'_beta_sum', aband["beta_sum{0}".format(k)])
         np.save(s1_angry[k]+'_delta_sum', aband["delta_sum{0}".format(k)])
         np.save(s1_angry[k]+'_gamma_sum', aband["gamma_sum{0}".format(k)])
         np.save(s1_angry[k]+'_theta_sum', aband["theta_sum{0}".format(k)])
         
         
    """ --------------------Separate Channels ------------------------------------"""
#    for v in range(0,7): #
#        for i in range(0,x):
#             
#             np.save(s1_happy[i]+'_alpha_{0}'.format(v), hband["alpha{0}".format(i)][v,:])
#             np.save(s1_happy[i]+'_beta_{0}'.format(v), hband["beta{0}".format(i)][v,:])
#             np.save(s1_happy[i]+'_delta_{0}'.format(v), hband["delta{0}".format(i)][v,:])
#             np.save(s1_happy[i]+'_gamma_{0}'.format(v), hband["gamma{0}".format(i)][v,:])
#             np.save(s1_happy[i]+'_theta_{0}'.format(v), hband["theta{0}".format(i)][v,:])
#             
#        for j in range(0,y):
#             np.save(s1_sad[j]+'_alpha_{0}'.format(v), sband["alpha{0}".format(j)][v,:])
#             np.save(s1_sad[j]+'_beta_{0}'.format(v), sband["beta{0}".format(j)][v,:])
#             np.save(s1_sad[j]+'_delta_{0}'.format(v), sband["delta{0}".format(j)][v,:])
#             np.save(s1_sad[j]+'_gamma_{0}'.format(v), sband["gamma{0}".format(j)][v,:])
#             np.save(s1_sad[j]+'_theta_{0}'.format(v), sband["theta{0}".format(j)][v,:])
#        
#        for k in range(0,z):
#             np.save(s1_angry[k]+'_alpha_{0}'.format(v), aband["alpha{0}".format(k)][v,:])
#             np.save(s1_angry[k]+'_beta_{0}'.format(v), aband["beta{0}".format(k)][v,:])
#             np.save(s1_angry[k]+'_delta_{0}'.format(v), aband["delta{0}".format(k)][v,:])
#             np.save(s1_angry[k]+'_gamma_{0}'.format(v), aband["gamma{0}".format(k)][v,:])
#             np.save(s1_angry[k]+'_theta_{0}'.format(v), aband["theta{0}".format(k)][v,:])
             
    """ ---------------------------PSD seperate -------------------------------------------"""
#    
#    for v in range(0,7):
#        for i in range(0,x):
#             np.save(s1_happy[i]+'_psd_{0}'.format(v), hpsd["hpsdx{0}".format(i)][v,:])
#        
#        for j in range(0,y):
#             np.save(s1_sad[j]+'_psd_{0}'.format(v), spsd["spsdx{0}".format(j)][v,:])
#             
#        
#        for k in range(0,z):
#            np.save(s1_angry[k]+'_psd_{0}'.format(v), apsd["apsdx{0}".format(k)][v,:])
    
    """ ========================== Absolute Power ============================ """
    
    """ take the abs_psd of each band, alpha, beta, delta, gamma and theta, and aggregate them into a vector that
        can be used as a feature for the classifier. """
    #
    #S01t1_happy_abspwr = Feature_calc.abs_psd_feature(S01t1_happy, Fs)
    #np.save('./Data/Training/ABS_PSD/S01t1_happy_W3O_abspwr',S01t1_happy_abspwr)
    
    hpwr = {}
    for i in range(0,x):
         hpwr["happy{0}".format(i)]=Feature_calc.abs_psd_feature(h["happy{0}".format(i)], Fs)
         np.save(s1_happy[i]+'_abspwr', hpwr["happy{0}".format(i)])
    spwr ={}
    for j in range(0,y):
         spwr["sad{0}".format(j)]=Feature_calc.abs_psd_feature(s["sad{0}".format(j)], Fs)
         np.save(s1_sad[j]+'_abspwr', spwr["sad{0}".format(j)])
         
    apwr = {}     
    for k in range(0,z):
        apwr["angry{0}".format(k)] = Feature_calc.abs_psd_feature(a["angry{0}".format(k)],Fs)
        np.save(s1_angry[k]+'_abspwr', apwr["angry{0}".format(k)])
         
    
    for i in range(0,x):
         hpwr["happy_sum{0}".format(i)]=np.sum(hpwr["happy{0}".format(i)],axis = 0)
         np.save(s1_happy[i]+'_abspwr_sum', hpwr["happy_sum{0}".format(i)])
    
    for j in range(0,y):
         spwr["sad_sum{0}".format(j)]=np.sum(spwr["sad{0}".format(j)],axis=0)
         np.save(s1_sad[j]+'_abspwr_sum', spwr["sad_sum{0}".format(j)])
    
    for k in range(0,z):
        apwr["angry_sum{0}".format(k)] = np.sum(apwr["angry{0}".format(k)],axis=0)
        np.save(s1_angry[k]+'_abspwr_sum', apwr["angry_sum{0}".format(k)])
         
    
    """=========================== PSD + Power ================================"""
    
    for i in range(0,x):
         hpwr["happy_PP{0}".format(i)]=np.hstack((hpwr["happy_sum{0}".format(i)],hpsd["hpsdx_sum{0}".format(i)]))
         np.save(s1_happy[i]+'_PP', hpwr["happy_PP{0}".format(i)])
    
    for j in range(0,y):
         spwr["sad_PP{0}".format(j)]=np.hstack((spwr["sad_sum{0}".format(j)],spsd["spsdx_sum{0}".format(j)]))
         np.save(s1_sad[j]+'_PP', spwr["sad_PP{0}".format(j)])
    
    for k in range(0,z):
    
        apwr["angry_PP{0}".format(k)] = np.hstack((apwr["angry_sum{0}".format(k)],apsd["apsdx_sum{0}".format(k)]))
        np.save(s1_angry[k]+'_PP', apwr["angry_PP{0}".format(k)])
    
    
    """ ======================= DLAT ================================================="""
    #hdlat = {}
    #for i in range(0,x):
    #     hdlat["happy{0}".format(i)]=Feature_calc.DLAT(hpsd["hpsdx{0}".format(i)], Fs)
    #     np.save(s1_happy[i]+'_dlatpsd', hdlat["happy{0}".format(i)])
    #sdlat = {}
    #for j in range(0,y):
    #     sdlat["sad{0}".format(j)]=Feature_calc.DLAT(spsd["spsdx{0}".format(j)], Fs)
    #     np.save(s1_sad[j]+'_dlatpsd', sdlat["sad{0}".format(j)])
    #     
    #adlat = {}
    #for k in range(0,z):
    #     adlat["angry{0}".format(k)]=Feature_calc.DLAT(apsd["apsdx{0}".format(k)], Fs)
    #     np.save(s1_angry[k]+'_dlatpsd', adlat["angry{0}".format(k)])
    #
    #     
    #     
    #"""================ PSD power and DLAT/DCAU ================================="""
    #     
    #     
    #for i in range(0,x):
    #     hpwr["happy_PPD{0}".format(i)]=np.hstack((hpwr["happy_PP{0}".format(i)],np.sum(hdlat["happy{0}".format(i)],axis=0)))
    #     np.save(s1_happy[i]+'_PPD', hpwr["happy_PPD{0}".format(i)])
    #
    #for j in range(0,y):
    #     spwr["sad_PPD{0}".format(j)]=np.hstack((spwr["sad_PP{0}".format(j)],np.sum(sdlat["sad{0}".format(j)],axis=0)))
    #     np.save(s1_sad[j]+'_PPD', spwr["sad_PPD{0}".format(j)])
    #
    #for k in range(0,z):
    #
    #    apwr["angry_PPD{0}".format(k)] = np.hstack((apwr["angry_PP{0}".format(k)],np.sum(adlat["angry{0}".format(k)],axis=0)))
    #    np.save(s1_angry[k]+'_PPD', apwr["angry_PPD{0}".format(k)])     
        
        
        
#    """=================== Valance and Arrousal ================================="""
#    hva = {}
#    hv = []
#    ha = []
#    for i in range(0,x):
#         hva["happyv{0}".format(i)]=Feature_calc.valence(h["happy{0}".format(i)], Fs)
#         np.save(s1_happy[i]+'_val', hva["happyv{0}".format(i)])
#         hva["happya{0}".format(i)]=Feature_calc.arrousal(h["happy{0}".format(i)], Fs)
#         np.save(s1_happy[i]+'_arrous', hva["happya{0}".format(i)])
#    
#         hv = np.append(hv,hva["happyv{0}".format(i)])
#    
#         ha = np.append(ha,hva["happya{0}".format(i)])
#         
#    sva = {}
#    sv = []
#    sa =[]
#    for j in range(0,y):
#         sva["sadv{0}".format(j)]=Feature_calc.valence(s["sad{0}".format(j)], Fs)
#         np.save(s1_sad[j]+'_val', sva["sadv{0}".format(j)])
#         sva["sada{0}".format(j)]=Feature_calc.arrousal(s["sad{0}".format(j)], Fs)
#         np.save(s1_sad[j]+'_arrous', sva["sada{0}".format(j)])
#         
#         sv = np.append(sv,sva["sadv{0}".format(j)])
#         
#         sa = np.append(sa,sva["sada{0}".format(j)])
#         
#    ava = {}
#    av =[]
#    avaa=[]
#    for k in range(0,z):
#         ava["angryv{0}".format(k)]=Feature_calc.valence(a["angry{0}".format(k)], Fs)
#         np.save(s1_angry[k]+'_val', ava["angryv{0}".format(k)])
#         ava["angrya{0}".format(k)]=Feature_calc.arrousal(a["angry{0}".format(k)], Fs)
#         np.save(s1_angry[k]+'_val', ava["angrya{0}".format(k)])
#         
#         
#         av = np.append(av, ava["angryv{0}".format(k)])
#         
#         avaa = np.append(avaa,ava["angrya{0}".format(k)] )
#         
#    Feature_calc.val_arr(av, avaa)
#    Feature_calc.val_arr(hv, ha)
#    
#    np.save('./Data/angryvalence',av)
#    np.save('./Data/angryarrousal',avaa)
#    np.save('./Data/happyvalence',hv)
#    np.save('./Data/happyarrousal',ha)
    
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
         hFD["happya{0}".format(i)], hFD["happyv{0}".format(i)]=Feature_calc.hfd_valarous(h["happy{0}".format(i)])
         aroushFD.append(hFD["happya{0}".format(i)])
         valhFD.append(hFD["happyv{0}".format(i)])
         hFD["happy{0}".format(i)]=Feature_calc.hfd_feat(h["happy{0}".format(i)])
         np.save(s1_happy[i]+'_HFD', hFD["happy{0}".format(i)])
         hFD["happy2{0}".format(i)]=Feature_calc.hfd_specfeat(h["happy{0}".format(i)])
         np.save(s1_happy[i]+'_HFDspec', hFD["happy2{0}".format(i)])
    
    for j in range(0,y):
         sFD["sada{0}".format(j)], sFD["sadv{0}".format(j)]=Feature_calc.hfd_valarous(s["sad{0}".format(j)])
         aroussFD.append(sFD["sada{0}".format(j)])
         valsFD.append(sFD["sadv{0}".format(j)])
         sFD["sad{0}".format(j)]=Feature_calc.hfd_feat(s["sad{0}".format(j)])
         np.save(s1_sad[j]+'_HFD', sFD["sad{0}".format(j)])
         sFD["sad2{0}".format(j)]=Feature_calc.hfd_specfeat(s["sad{0}".format(j)])
         np.save(s1_sad[j]+'_HFDspec', sFD["sad2{0}".format(j)])
    
    for k in range(0,z):
         aFD["angrya{0}".format(k)], aFD["angryv{0}".format(k)]=Feature_calc.hfd_valarous(a["angry{0}".format(k)])
         arousaFD.append(aFD["angrya{0}".format(k)])
         valaFD.append(aFD["angryv{0}".format(k)])
         aFD["angry{0}".format(k)]=Feature_calc.hfd_feat(a["angry{0}".format(k)])
         np.save(s1_angry[k]+'_HFD', aFD["angry{0}".format(k)])
         aFD["angry2{0}".format(k)]=Feature_calc.hfd_specfeat(a["angry{0}".format(k)])
         np.save(s1_angry[k]+'_HFDspec', aFD["angry2{0}".format(k)])
         
    valtest = np.append(valaFD,valhFD)
    varoustest = np.append(arousaFD,aroushFD)
    print 'angry min %f max %f' %(np.min(valaFD), np.max(valaFD))
    print 'happy min %f max %f' %( np.min(valhFD), np.max(valhFD))
