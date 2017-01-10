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
from sklearn.preprocessing import MultiLabelBinarizer
#
#iris = load_iris()
#
#X, y = iris.data, iris.target
#

WS = 10 # window size variable
s1_happy = np.load('./Data/Training/happyW{0}.npy'.format(WS))
s1_angry = np.load('./Data/Training/angryW{0}.npy'.format(WS))
s1_norm = np.load('./Data/Training/normW{0}.npy'.format(WS))
#s1_sad = np.load('./Data/Training/sadW{0}.npy'.format(WS))

#s1_happy = np.append(s1_happy, np.load('./Data/Training/happyW3O.npy'))
#
#s1_angry = np.append(s1_angry,np.load('./Data/Training/angryW3O.npy'))
#
#s1_norm = np.append(s1_norm, np.load('./Data/Training/normW3O.npy'))

x = np.size(s1_happy)
y = np.size(s1_norm)
z = np.size(s1_angry)
h = {}
s = {}
a = {}
out = []
acc = []
ker = []
ff=[]


#win = 10 # window size

feature = ['_gamma_sum.npy','_theta_sum.npy','_psd_sum.npy','_psd.npy','_PP.npy','_PPD.npy','_alpha_sum.npy','_psd_flat.npy','_HFD.npy','_HFDspec.npy','_psd_mean.npy','_beta_sum.npy','_delta_sum.npy','_abspwr.npy','_abspwr_sum.npy','_theta_gamma.npy'] #'_PSD2.npy','_abspwr.npy','_abspwr_sum.npy','_alpha_0.npy','_alpha_1.npy','_alpha_2.npy','_alpha_3.npy','_alpha_4.npy','_alpha_5.npy','_alpha_6.npy','_alpha_7.npy','_beta_0.npy','_beta_1.npy','_beta_2.npy','_beta_3.npy','_beta_4.npy','_beta_5.npy','_beta_6.npy','_beta_7.npy','_delta_0.npy','_delta_1.npy','_delta_2.npy','_delta_3.npy','_delta_4.npy','_delta_5.npy','_delta_6.npy','_delta_7.npy','_gamma_0.npy','_gamma_1.npy','_gamma_2.npy','_gamma_3.npy','_gamma_4.npy','_gamma_5.npy','_gamma_6.npy','_gamma_7.npy','_theta_0.npy','_theta_1.npy','_theta_2.npy','_theta_3.npy','_theta_4.npy','_theta_5.npy','_theta_6.npy','_theta_7.npy','_psd_0.npy','_psd_1.npy','_psd_2.npy','_psd_3.npy','_psd_4.npy','_psd_5.npy','_psd_6.npy','_psd_7.npy'
p=1
""" ------------0--------------1-----------------2-------------3-----------4-----------5--------------6-------------7-----------------8----------------9----------10-----------11------------12----------13-----------14"""

knn = 7
clf = KNeighborsClassifier(n_neighbors=knn)
   
#clf = svm.SVC(kernel = 'linear', C =3.0, decision_function_shape='ovr')
#

#clf = svm.SVC(kernel = 'rbf', C =3.0, decision_function_shape='ovr')

#clf = svm.SVC(kernel = 'poly', C =1.0, degree = 3, decision_function_shape='ovr')
#       
""" -------------------Features---------------------------------------------------------------------------"""

    
    
for i in range(0,x):
    h["happy{0}".format(i)]=np.load(s1_happy[i]+feature[p])
# Get Sad
for j in range(0,y):
    s["sad{0}".format(j)]=np.load(s1_norm[j]+feature[p])
    
for k in range(0,z):
    a["angry{0}".format(k)] = np.load(s1_angry[k]+feature[p])

"""-----------------Stack the Data ---------------------------------------------"""

train = np.vstack((h['happy0'],h['happy1']))

for k in range(2,x):
    train = np.vstack((train,h["happy{0}".format(k)]))
htrain,zz=np.shape(train)
yh =np.ones(htrain)

#        
for m in range(0,z):
    train = np.vstack((train,a["angry{0}".format(m)]))
atrain,zz=np.shape(train)
#        ya = np.ones(atrain-strain)*2
ya = np.ones(atrain-htrain)*2
#
for l in range(0,y):
    train = np.vstack((train,s["sad{0}".format(l)]))
strain,zz =np.shape(train)
ys = np.zeros(strain-atrain)


train = np.array(train)
train.reshape(-1,1)



yl = np.append(yh,ya)
yl = np.append(yl,ys)
yl.reshape(-1,1)


X_train, X_test, Y_train, Y_test = train_test_split(train, yl, test_size=0.33, random_state=42)

clf.fit(X_train,Y_train)


accuracy = 0.0

result = clf.predict(X_test)

for m in range(0,np.size(result)):
#        print 'Sample Lable', Y_test[m] ,'sample prediction', s[m]
    if Y_test[m] == result[m]:
        accuracy = accuracy +1.0
        
accuracy = accuracy/np.size(result)*100
print accuracy

"""========================= Train Classifier Fully =========================="""
clf.fit(train,yl)
joblib.dump(clf, './Data/Training/Live/17Nov/HvAvN_W{0}{1}_KNN{3}_clf_{2}.pkl'.format(WS,feature[p],accuracy, knn) ) 
a = "./Data/Training/Live/17Nov/HvAvN_W{0}{1}_KNN{3}_clf_{2}.pkl".format(WS,feature[p],accuracy,knn)