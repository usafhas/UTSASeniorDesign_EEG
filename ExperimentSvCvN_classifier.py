# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 17:08:34 2016

@author: Heath
"""
import numpy as np
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
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

WS = windows[1] # window size variable

#    s1_happy = np.load('./Data/Training/happyW{0}.npy'.format(WS))
s1_stress = np.load('./Data/Experiments/stressW{0}.npy'.format(WS))
s1_norm = np.load('./Data/Experiments/normW{0}.npy'.format(WS))
s1_calm = np.load('./Data/Experiments/calmW{0}.npy'.format(WS))

x = np.size(s1_norm)
y = np.size(s1_calm)
z = np.size(s1_stress)

feature = ['_gamma_sum.npy','_theta_sum.npy','_psd_sum.npy','_psd.npy','_PP.npy','_alpha_sum.npy','_psd_flat.npy','_HFD.npy','_psd_mean.npy','_delta_sum.npy','_abspwr.npy','_abspwr_sum.npy','_cog.npy','_spec.npy','_abspwr_alpha8.npy']#,'_theta_gamma.npy' #'_PSD2.npy','_abspwr.npy','_abspwr_sum.npy','_alpha_0.npy','_alpha_1.npy','_alpha_2.npy','_alpha_3.npy','_alpha_4.npy','_alpha_5.npy','_alpha_6.npy','_alpha_7.npy','_beta_0.npy','_beta_1.npy','_beta_2.npy','_beta_3.npy','_beta_4.npy','_beta_5.npy','_beta_6.npy','_beta_7.npy','_delta_0.npy','_delta_1.npy','_delta_2.npy','_delta_3.npy','_delta_4.npy','_delta_5.npy','_delta_6.npy','_delta_7.npy','_gamma_0.npy','_gamma_1.npy','_gamma_2.npy','_gamma_3.npy','_gamma_4.npy','_gamma_5.npy','_gamma_6.npy','_gamma_7.npy','_theta_0.npy','_theta_1.npy','_theta_2.npy','_theta_3.npy','_theta_4.npy','_theta_5.npy','_theta_6.npy','_theta_7.npy','_psd_0.npy','_psd_1.npy','_psd_2.npy','_psd_3.npy','_psd_4.npy','_psd_5.npy','_psd_6.npy','_psd_7.npy'
#        feature = ['_gamma_sum.npy','_theta_sum.npy','_psd_sum.npy','_psd.npy','_PP.npy','_PPD.npy','_alpha_sum.npy','_psd_flat.npy','_HFD.npy','_HFDspec.npy','_psd_mean.npy','_beta_sum.npy','_delta_sum.npy','_abspwr.npy','_abspwr_sum.npy','_theta_gamma.npy','_alpha_0.npy','_alpha_1.npy','_alpha_2.npy','_alpha_3.npy','_alpha_4.npy','_alpha_5.npy','_alpha_6.npy','_alpha_7.npy','_beta_0.npy','_beta_1.npy','_beta_2.npy','_beta_3.npy','_beta_4.npy','_beta_5.npy','_beta_6.npy','_beta_7.npy','_delta_0.npy','_delta_1.npy','_delta_2.npy','_delta_3.npy','_delta_4.npy','_delta_5.npy','_delta_6.npy','_delta_7.npy','_gamma_0.npy','_gamma_1.npy','_gamma_2.npy','_gamma_3.npy','_gamma_4.npy','_gamma_5.npy','_gamma_6.npy','_gamma_7.npy','_theta_0.npy','_theta_1.npy','_theta_2.npy','_theta_3.npy','_theta_4.npy','_theta_5.npy','_theta_6.npy','_theta_7.npy','_psd_0.npy','_psd_1.npy','_psd_2.npy','_psd_3.npy','_psd_4.npy','_psd_5.npy','_psd_6.npy','_psd_7.npy'] #'_PSD2.npy'

""" ------------0--------------1-----------------2-------------3-----------4-----------5--------------6-------------7-----------------8----------------9----------10-----------11------------12----------13-----------14"""

#clf = KNeighborsClassifier(n_neighbors=5) 
#
#
#clf = KNeighborsClassifier(n_neighbors=3) 
#
#clf = KNeighborsClassifier(n_neighbors=7)
#
#clf = svm.SVC(kernel = 'rbf', C =3.0, decision_function_shape='ovr')
#
#clf = naive_bayes.GaussianNB()
#
#clf = QuadraticDiscriminantAnalysis()
#
#clf = svm.SVC(kernel = 'poly', C =1.0, degree = 5, decision_function_shape='ovr')

    
""" -------------------Features---------------------------------------------------------------------------"""
q=1
clf = QuadraticDiscriminantAnalysis()
        
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


"""========================= Train Classifier Fully =========================="""
clf.fit(train,yl)
joblib.dump(clf, './Classifier_Test/classifiers/SvCvN_W{0}{1}_QDA.pkl'.format(WS,feature[q]) ) 