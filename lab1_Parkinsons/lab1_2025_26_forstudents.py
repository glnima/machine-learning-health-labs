# -*- coding: utf-8 -*-
"""

@author: Monica Visintin

Regress Total UPDRS from the other features in file "parkinsons_updrs_av.csv"

"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

pd.set_option('display.precision', 3)
#%% Read the input csv file
plt.close('all') # close all the figures that might still be open from previous runs
X=pd.read_csv("parkinsons_updrs_av.csv") # read the dataset; x is a Pandas dataframe
features=list(X.columns)#  list of features in the dataset
subj=pd.unique(X['subject#'])# existing values of patient ID
print("The original dataset shape  is ",X.shape)
print("The number of distinct patients in the dataset is ",len(subj))
print("the original dataset features are ",len(features))
print(features)
Np,Nc=X.shape# Np = number of rows/ptients Nc=number Nf of regressors + 1 (regressand total UPDRS is included)
#%% Have a look at the dataset
print(X.describe().T) # gives the statistical description of the content of each column
print(X.info())
#%% Measure and show the covariance matrix
Xnorm=(X-X.mean())/X.std()# normalized/standardized/scaled data
c=Xnorm.cov()# note: xx.cov() gives a result that cannot be visually read
plt.figure()
plt.matshow(np.abs(c.values),fignum=0)# absolute value of corr.coeffs
plt.xticks(np.arange(len(features)), features, rotation=90)
plt.yticks(np.arange(len(features)), features, rotation=0)    
plt.colorbar()
plt.title('Correlation coefficients of the features')
plt.tight_layout()
plt.savefig('./corr_coeff.png') # save the figure
plt.draw()
plt.figure()
c.total_UPDRS.plot()
plt.grid()
plt.xticks(np.arange(len(features)),features,rotation=90)
plt.title('Corr. coeff. between total_UPDRS and the other features')
plt.tight_layout()
plt.draw()
plt.savefig('./UPDRS_corr_coeff.png') # save the figure
#%% Shuffle the data (two out of many methods)
# first method:
np.random.seed(30) # set the seed for random shuffling
# indexsh=np.arange(Np) # generate array [0,1,...,Np-1]
# np.random.shuffle(indexsh) # shuffle the array
# Xsh=X.copy()
# Xsh=Xsh.set_axis(indexsh,axis=0,inplace=False) # shuffle accordingly the dataframe
# Xsh=Xsh.sort_index(axis=0) # reset index of the dataframe
# comment: Xsh.reset_index() exists, but a further index column would be created
# second method
Xsh=X.sample(frac=1, replace=False, random_state=30, axis=0, ignore_index=True)
#Xsh=X.sample(frac=1, replace=False, axis=0, ignore_index=True)

#%% Start working on training and test subsets
Ntr=int(Np*0.5)  # number of training points
Nte=Np-Ntr   # number of test points
#%% evaluate mean and st.dev. for the training data only
X_tr=Xsh[0:Ntr]# dataframe that contains only the training data
mm=X_tr.mean()# mean (series) of each feature
ss=X_tr.std()# standard deviation (series) of each feature
my=mm['total_UPDRS']# mean of regressand/total UPDRS (for later use)
sy=ss['total_UPDRS']# st.dev of regressand/total UPDRS (for later use)
#%% Generate the normalized/scaled training and test datasets, remove unwanted regressors
Xsh_norm=(Xsh-mm)/ss# normalized data
ysh_norm=Xsh_norm['total_UPDRS']# regressand only
Xsh_norm=Xsh_norm.drop(['total_UPDRS','subject#'],axis=1)# regressors only
regressors=list(Xsh_norm.columns)
Nf = len(regressors) # number of regressors
print("The new regressors are: ",len(regressors))
#print(regressors)
Xsh_norm=Xsh_norm.values # from dataframe to Ndarray
ysh_norm=ysh_norm.values # from dataframe to Ndarray
X_tr_norm=Xsh_norm[0:Ntr] # regressors for training phase
X_te_norm=Xsh_norm[Ntr:] # regressors for test phase
y_tr_norm=ysh_norm[0:Ntr] # regressand for training phase
y_te_norm=ysh_norm[Ntr:] #regressand for test phase
print(X_tr_norm.shape,X_te_norm.shape)
#%% LLS regression
w_hat=np.linalg.inv(X_tr_norm.T@X_tr_norm)@(X_tr_norm.T@y_tr_norm)
y_hat_tr_norm=X_tr_norm@w_hat
y_hat_te_norm=X_te_norm@w_hat
#%% de-normalize data realted to total-UPDRS (the regressand)
y_tr=y_tr_norm*sy+my
y_te=y_te_norm*sy+my
y_hat_tr=y_hat_tr_norm*sy+my
y_hat_te=y_hat_te_norm*sy+my
#%% plot the optimum weight vector for LLS
nn=np.arange(Nf)
plt.figure(figsize=(6,4))
plt.plot(nn,w_hat,'-o')
ticks=nn
plt.xticks(ticks, regressors, rotation=90)
plt.ylabel(r'$\^w(n)$')
plt.title('LLS-Optimized weights')
plt.grid()
plt.tight_layout()
plt.savefig('./LLS-what.png')
plt.draw()
#%% plot the error histograms
E_tr=(y_tr-y_hat_tr)# errors on the training dataset
E_te=(y_te-y_hat_te)# errors on the test dataset
M=np.max([np.max(E_tr),np.max(E_te)])
m=np.min([np.min(E_tr),np.min(E_te)])
common_bins=np.arange(m,M,(M-m)/50)
e=[E_tr,E_te]
plt.figure(figsize=(6,4))
plt.hist(e,bins=common_bins,density=True, histtype='bar',label=['training','test'])
plt.xlabel(r'$e=y-\^y$')
plt.ylabel(r'$P(e$ in bin$)$')
plt.legend()
plt.grid()
plt.title('LLS-Error histograms using all the training dataset')
plt.tight_layout()
plt.savefig('./LLS-hist.png')
plt.draw()
#%% plot the regression lines
plt.figure(figsize=(4,4))
plt.plot(y_te,y_hat_te,'.',label='all')
plt.legend()
v=plt.axis()
plt.plot([v[0],v[1]],[v[0],v[1]],'r',linewidth=2)
plt.xlabel(r'$y$')
plt.axis('square')
plt.ylabel(r'$\^y$')
plt.grid()
plt.title('LLS-test')
plt.tight_layout()
plt.savefig('./LLS-yhat_vs_y.png')
plt.draw()


#%% statistics of the errors

# on the training dataset:
E_tr_max=E_tr.max()# max error
E_tr_min=E_tr.min()# min error
E_tr_mu=E_tr.mean()# mean error
E_tr_sig=E_tr.std()# standard deviation of the error
E_tr_MSE=np.mean(E_tr**2)# mean square error
R2_tr=1-E_tr_MSE/(np.var(y_tr))# coefficient of determination
# correlation coefficient:
c_tr=np.mean((y_tr-y_tr.mean())*(y_hat_tr-y_hat_tr.mean()))/(y_tr.std()*y_hat_tr.std())
# on the test dataset
E_te_max=E_te.max()# max error
E_te_min=E_te.min()# min error
E_te_mu=E_te.mean()# mean error
E_te_sig=E_te.std()# standard deviation of the error
E_te_MSE=np.mean(E_te**2)# mean square error
R2_te=1-E_te_MSE/(np.var(y_te))# coefficient of determination
# correlation coefficient:
c_te=np.mean((y_te-y_te.mean())*(y_hat_te-y_hat_te.mean()))/(y_te.std()*y_hat_te.std())
cols=['min','max','mean','std','MSE','R^2','corr_coeff']
rows=['Training','test']
p=np.array([
    [E_tr_min,E_tr_max,E_tr_mu,E_tr_sig,E_tr_MSE,R2_tr,c_tr],
    [E_te_min,E_te_max,E_te_mu,E_te_sig,E_te_MSE,R2_te,c_te],
            ])

results=pd.DataFrame(p,columns=cols,index=rows)
print(results)
plt.show()
