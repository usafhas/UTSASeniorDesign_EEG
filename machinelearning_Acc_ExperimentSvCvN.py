# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 17:08:34 2016

@author: Heath
"""
import numpy as np
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import Feature_calc # calculate features from LSL
from sklearn.externals import joblib
from sklearn.cross_validation import train_test_split
import csv
from sklearn import naive_bayes
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import cross_val_score # kstratifiedcrossfold method for training data


windows = [10, 15, 30]
n = {}
a = {}
c = {}
out = []
acc = []
csva = []
ker = []
ff=[]


for bbb in range(0,np.size(windows)):

    WS = windows[bbb] # window size variable
    
#    s1_happy = np.load('./Data/Training/happyW{0}.npy'.format(WS))
    s1_stress = np.load('./Data/Experiments/stressW{0}.npy'.format(WS))
    s1_norm = np.load('./Data/Experiments/normW{0}.npy'.format(WS))
    s1_calm = np.load('./Data/Experiments/calmW{0}.npy'.format(WS))
    
    
    x = np.size(s1_norm)
    y = np.size(s1_calm)
    z = np.size(s1_stress)


    acc.append("\r\n\r\n\r\n=====================Stress vs Calm %s====================================== \r\n" %WS)
    print "\r\n===================Stressed vs calm %s============================ \r\n"%WS
    hvan = "Stress v Calm v Nuetral"
    feature = ['_gamma_sum.npy','_theta_sum.npy','_psd_sum.npy','_psd.npy','_PP.npy','_alpha_sum.npy','_psd_flat.npy','_HFD.npy','_psd_mean.npy','_delta_sum.npy','_abspwr.npy','_abspwr_sum.npy','_cog.npy','_spec.npy']#,'_theta_gamma.npy' #'_PSD2.npy','_abspwr.npy','_abspwr_sum.npy','_alpha_0.npy','_alpha_1.npy','_alpha_2.npy','_alpha_3.npy','_alpha_4.npy','_alpha_5.npy','_alpha_6.npy','_alpha_7.npy','_beta_0.npy','_beta_1.npy','_beta_2.npy','_beta_3.npy','_beta_4.npy','_beta_5.npy','_beta_6.npy','_beta_7.npy','_delta_0.npy','_delta_1.npy','_delta_2.npy','_delta_3.npy','_delta_4.npy','_delta_5.npy','_delta_6.npy','_delta_7.npy','_gamma_0.npy','_gamma_1.npy','_gamma_2.npy','_gamma_3.npy','_gamma_4.npy','_gamma_5.npy','_gamma_6.npy','_gamma_7.npy','_theta_0.npy','_theta_1.npy','_theta_2.npy','_theta_3.npy','_theta_4.npy','_theta_5.npy','_theta_6.npy','_theta_7.npy','_psd_0.npy','_psd_1.npy','_psd_2.npy','_psd_3.npy','_psd_4.npy','_psd_5.npy','_psd_6.npy','_psd_7.npy'
#        feature = ['_gamma_sum.npy','_theta_sum.npy','_psd_sum.npy','_psd.npy','_PP.npy','_PPD.npy','_alpha_sum.npy','_psd_flat.npy','_HFD.npy','_HFDspec.npy','_psd_mean.npy','_beta_sum.npy','_delta_sum.npy','_abspwr.npy','_abspwr_sum.npy','_theta_gamma.npy','_alpha_0.npy','_alpha_1.npy','_alpha_2.npy','_alpha_3.npy','_alpha_4.npy','_alpha_5.npy','_alpha_6.npy','_alpha_7.npy','_beta_0.npy','_beta_1.npy','_beta_2.npy','_beta_3.npy','_beta_4.npy','_beta_5.npy','_beta_6.npy','_beta_7.npy','_delta_0.npy','_delta_1.npy','_delta_2.npy','_delta_3.npy','_delta_4.npy','_delta_5.npy','_delta_6.npy','_delta_7.npy','_gamma_0.npy','_gamma_1.npy','_gamma_2.npy','_gamma_3.npy','_gamma_4.npy','_gamma_5.npy','_gamma_6.npy','_gamma_7.npy','_theta_0.npy','_theta_1.npy','_theta_2.npy','_theta_3.npy','_theta_4.npy','_theta_5.npy','_theta_6.npy','_theta_7.npy','_psd_0.npy','_psd_1.npy','_psd_2.npy','_psd_3.npy','_psd_4.npy','_psd_5.npy','_psd_6.npy','_psd_7.npy'] #'_PSD2.npy'
    q=0
    """ ------------0--------------1-----------------2-------------3-----------4-----------5--------------6-------------7-----------------8----------------9----------10-----------11------------12----------13-----------14"""
    for p in range(0,6):
        if p ==0:
            clf = KNeighborsClassifier(n_neighbors=5) 
            print "=====================using KNN = 5================================="
            cs = "using KNN = 5"
        elif p ==1:
            clf = KNeighborsClassifier(n_neighbors=3) 
            print "============================using KNN = 3========================="
            cs = "using KNN = 3"
        elif p==2:
            clf = KNeighborsClassifier(n_neighbors=7)
            print "================================using KNN = 7========================="
            cs = "using KNN = 7"
        elif p==3:
            clf = svm.SVC(kernel = 'rbf', C =3.0, decision_function_shape='ovr')
            print "===============================using Gaussian============================="
            cs = "using Gaussian"
        elif p ==4:
            clf = naive_bayes.GaussianNB()
            print "=====================Naive Bays Gaussian==============================="
            cs = "Naive Bayes Gaussian"
        elif p ==5:
            clf = QuadraticDiscriminantAnalysis()
            print "=====================Quadratic Discriminant Analysis==============================="
            cs = "Quadratic Discriminant Analysis"
        else:
            clf = svm.SVC(kernel = 'poly', C =1.0, degree = 5, decision_function_shape='ovr')
            print "=====================using Poly degree = 3==============================="
            cs = "using Poly degree = 5"
            
        """ -------------------Features---------------------------------------------------------------------------"""
        for q in range(0,np.size(feature)):
#            if (p ==4 or p==5 or p==6) and q == 4:
#                q+=1
#            if (p ==4 or p==5 or p==6) and q ==5:
#                q+=1
#            if (p ==4 or p==5 or p==6) and q == 12:
#                q+=1
#            if (p ==4 or p==5 or p==6) and q ==13:
#                q+=1
#                break
                
            print "feature %s" %feature[q]
    
            for i in range(0,x):
                n["norm{0}".format(i)]=np.load(s1_norm[i]+feature[q])
            # Get Sad
            for j in range(0,y):
                c["calm{0}".format(j)]=np.load(s1_calm[j]+feature[q])
                
            for k in range(0,z):
                a["angry{0}".format(k)] = np.load(s1_stress[k]+feature[q])
                
            sizefx= np.shape(np.load(s1_norm[0]+feature[q]))
            
            """-----------------Stack the Data ---------------------------------------------"""
            
            train = np.vstack((a['angry0'],a['angry1']))
            
            for k in range(2,z):
                train = np.vstack((train,a["angry{0}".format(k)]))
            htrain,zz=np.shape(train)
            yh =np.ones(htrain)
            
            #        
            for m in range(0,y):
                train = np.vstack((train,c["calm{0}".format(m)]))
            atrain,zz=np.shape(train)
            #        ya = np.ones(atrain-strain)*2
            ya = np.ones(atrain-htrain)*2
            #
   
            for l in range(0,x):
                 train = np.vstack((train,n["norm{0}".format(l)]))
            ntrain,zz = np.shape(train)
            yn = np.ones(ntrain-atrain)*0
                 
                 
            train = np.array(train)
            train.reshape(-1,1)
            
            
            
            yl = np.append(yh,ya)
            yl = np.append(yl,yn)
            yl.reshape(-1,1)
            
            
            scores = cross_val_score(clf, train, yl, cv=3)
            accuracy = np.mean(scores)
            print accuracy
            if accuracy >= 80:
                acstr = "\r\n%s \t\t\t %s \t\t \\b %d \\b0\r\n" %(cs, feature[q], accuracy)
            else:
                acstr = "\r\n%s \t\t\t %s \t\t %d \r\n" %(cs, feature[q], accuracy)
            print acstr
            acc.append(acstr)
            if np.size(sizefx) ==2:
                csva.append(['\t',cs, '\t',feature[q],'\t', accuracy,'\t',hvan,'\t',WS,'\t', "x=",sizefx[0]," y=",sizefx[1], '\t'])
            else:
                csva.append(['\t',cs, '\t',feature[q],'\t', accuracy,'\t',hvan,'\t',WS,'\t', "x=",1, "y=",sizefx[0], '\t'])
            

"""========================= Train Classifier Fully =========================="""
#clf.fit(train,yl)
#joblib.dump(clf, './Data/Training/Live/HvAvN_W3_PSD_KNN5_clf_69.pkl')  
with open('accuracy.txt', 'w') as f:
    f.writelines(acc)
with open('accuracy.rtf', 'w') as f:
    f.write(r'{\rtf1\ansi\ansicpg1252\deff0\deflang1033{\fonttbl{\f0\fswiss\fcharset0 Arial;}}')
    f.writelines(acc)
    f.write(r'}\n\x00')

with open('Accuracy_testing_experiment_SvNvC.csv', 'wb') as myfile:
    wr = csv.writer(myfile, delimiter = '\t', quoting=csv.QUOTE_NONE)
    x,y = np.shape(csva)
    for c in range (0,x):

        wr.writerow([csva[c]])
#    wr.writerows(csva)  