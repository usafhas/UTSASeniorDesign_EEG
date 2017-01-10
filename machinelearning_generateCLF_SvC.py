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
from sklearn import naive_bayes
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import cross_val_score # for kstratified cross validation algorithm

WS = 30 # window size variable
s1_happy = np.load('./Data/Training/happyW{0}.npy'.format(WS))
s1_angry = np.load('./Data/Training/angryW{0}.npy'.format(WS))
s1_norm = np.load('./Data/Training/normW{0}.npy'.format(WS))
s1_calm = np.load('./Data/Training/calmW{0}.npy'.format(WS))
#s1_sad = np.load('./Data/Training/sadW{0}.npy'.format(WS))

#s1_happy = np.append(s1_happy, np.load('./Data/Training/happyW3O.npy'))
#
#s1_angry = np.append(s1_angry,np.load('./Data/Training/angryW3O.npy'))
#
#s1_norm = np.append(s1_norm, np.load('./Data/Training/normW3O.npy'))

x = np.size(s1_happy)
y = np.size(s1_calm)
z = np.size(s1_angry)
h = {}
s = {}
a = {}
c = {}
out = []
acc = []
ker = []
ff=[]


#win = 10 # window size

feature = ['_gamma_sum.npy','_theta_sum.npy','_psd_sum.npy','_psd.npy','_PP.npy','_alpha_sum.npy','_psd_flat.npy','_HFD.npy','_psd_mean.npy','_delta_sum.npy','_abspwr.npy','_abspwr_sum.npy','_cog.npy','_spec.npy']
""" -----------0 ------------------1 ------------2--------------3-----------4---------5---------------6--------------7-----------8--------------9-----------------10------------11-------------12---------------13"""
p=7
knn=5
#clf = KNeighborsClassifier(n_neighbors=knn) 

#clf = KNeighborsClassifier(n_neighbors=3) 
#
#clf = KNeighborsClassifier(n_neighbors=7)
#
#clf = svm.SVC(kernel = 'rbf', C =3.0, decision_function_shape='ovr')
#
#clf = naive_bayes.GaussianNB()
#
clf = QuadraticDiscriminantAnalysis()
#
#clf = svm.SVC(kernel = 'poly', C =1.0, degree = 5, decision_function_shape='ovr')

    
    
for i in range(0,x):
    h["happy{0}".format(i)]=np.load(s1_happy[i]+feature[p])
# Get Sad
for j in range(0,y):
    c["calm{0}".format(j)]=np.load(s1_calm[j]+feature[p])
    
for k in range(0,z):
    a["angry{0}".format(k)] = np.load(s1_angry[k]+feature[p])

"""-----------------Stack the Data ---------------------------------------------"""

train = np.vstack((a['angry0'],a['angry1']))
			
for k in range(2,z):
	train = np.vstack((train,a["angry{0}".format(k)]))
htrain,zz=np.shape(train)
yh =np.ones(htrain)
        
for m in range(0,y):
	train = np.vstack((train,c["calm{0}".format(m)]))
atrain,zz=np.shape(train)
ya = np.ones(atrain-htrain)*2

train = np.array(train)
train.reshape(-1,1)
	
yl = np.append(yh,ya)
yl.reshape(-1,1)


#X_train, X_test, Y_train, Y_test = train_test_split(train, yl, test_size=0.33, random_state=42)

scores = cross_val_score(clf, X_train, Y_train) # kstratified fold cross validation, automatically splits data set up into 3 parts by default
print scores
print np.mean(scores)

#clf.fit(X_train,Y_train)


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
joblib.dump(clf, './Data/Classifier/Stress_Calm/SvC_W{0}{1}_KNN{3}_clf_{2}.pkl'.format(WS,feature[p],accuracy, knn) ) 
a = "./Data/Classifier/Stress_Calm/SvC_W{0}{1}_KNN{3}_clf_{2}.pkl".format(WS,feature[p],accuracy,knn)