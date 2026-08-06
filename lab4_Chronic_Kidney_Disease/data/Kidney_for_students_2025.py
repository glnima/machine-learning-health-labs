import pandas as pd
import numpy as np
from sklearn import tree
import matplotlib.pyplot as plt
plt.close('all')
#%%
# define the feature names:
feat_names=['age','bp','sg','al','su','rbc','pc',
'pcc','ba','bgr','bu','sc','sod','pot','hemo',
'pcv','wbcc','rbcc','htn','dm','cad','appet','pe',
'ane','classk']
feat_cat=np.array(['num','num','cat','cat','cat','cat','cat','cat','cat',
         'num','num','num','num','num','num','num','num','num',
         'cat','cat','cat','cat','cat','cat','cat'])
# import the dataframe:
#xx=pd.read_csv("./Chronic_Kidney_Disease/chronic_kidney_disease.arff",sep=',',
#               skiprows=29,names=feat_names, 
#               header=None,na_values=['?','\t?'],
#               warn_bad_lines=True)
xx=pd.read_csv("./data/chronic_kidney_disease_v2.arff",sep=',',
    skiprows=29,names=feat_names, 
    header=None,na_values=['?','\t?'],)
Np,Nf=xx.shape
#%% change categorical data into numbers:
target_names = ['notckd','ckd']# for the final plots
mapping={
    'normal':0,
    'abnormal':1,
    'present':1,
    'notpresent':0,
    'yes':1,
    ' yes':1,
    'no':0,
    '\tno':0,
    '\tyes':1,
    'ckd':1,
    'notckd':0,
    'poor':1,
    'good':0,
    'ckd\t':1}
xx=xx.replace(mapping.keys(),mapping.values())

print('cardinality of each feature:')
print(xx.nunique())# show the cardinality of each feature in the dataset; in particular classk should have only two possible values
#%%
miss_values = xx.isnull().sum(axis=1)
for k in range(miss_values.max()+1):
    print(k,np.sum(miss_values==k))
#%% manage the missing data through regression
print(xx.info())
x=xx.copy()
# drop rows with less than 19=Nf-6 recorded features:
x=x.dropna(thresh=19)
x.reset_index(drop=True, inplace=True)# necessary to have index without "jumps"
n=x.isnull().sum(axis=1)# check the number of missing values in each row
print('Number of points in the original dataset: ',xx.shape[0])
print('reduced dataset: at least 19 values per row')
print('number of points in the reduced dataset: ',x.shape[0])
print('max number of missing values in the reduced dataset: ',n.max())
# take the rows with exctly Nf=25 useful features; this is going to be the training dataset
# for regression
Xtrain=x.dropna(thresh=25)
Xtrain.reset_index(drop=True, inplace=True)# reset the index of the dataframe
print('Number of points in the training dataset: ',Xtrain.shape[0])
#%% normalization
XtrainNp = Xtrain.values # Numpy 2D array
mm=XtrainNp.mean(axis=0)
ss=XtrainNp.std(axis=0)
XtrainNp_norm=(XtrainNp-mm)/ss
#% normalize the entire dataset using the coeffs found for the training dataset
X_normNp=(x.values-mm)/ss
Np,Nf=X_normNp.shape
#%% run linear regression using least squares on all the missing data
for kk in range(Np):
    xrow=X_normNp[kk,:]#k-th row
    mask=np.isnan(xrow)# columns with nan in row k
    Data_tr_norm=XtrainNp_norm[:,~mask]# remove the columns from the training dataset
    y_tr_norm=XtrainNp_norm[:,mask]# columns to be regressed
    w1=np.linalg.inv(np.dot(Data_tr_norm.T,Data_tr_norm))
    w=np.dot(np.dot(w1,Data_tr_norm.T),y_tr_norm)# weight vector
    ytest_norm=np.dot(xrow[~mask],w)
    xrow[mask]=ytest_norm 
    X_normNp[kk]=xrow# substitute nan with regressed values
x_new=X_normNp*ss+mm # denormalize
#%% manage categorical features
# get the possible values (i.e. alphabet) for the categorical features
alphabets=[]
for k in range(len(feat_cat)):
    if feat_cat[k]=='cat':
        val=Xtrain[Xtrain.columns[k]].unique()
        alphabets.append(np.sort(val))
    else:
        alphabets.append('num')
index=np.argwhere(feat_cat=='cat').flatten()
for k in index:
    val=alphabets[k].flatten()
    c=x_new[:,k]
    val = val.reshape(1,-1) # force row vector
    c=c.reshape(-1,1) # force column vector
    d=(val-c)**2 # find the square distances
    ii=d.argmin(axis=1) # find the closest categorical value
    cc=val[0,ii] # cc contains only the categorical values
    x_new[:,k]=cc
#
x_new= pd.DataFrame(x_new,columns=feat_names) # go back to Pandas dataframe
#%% check the distributions
L=x_new.shape[0]
plotCDF=True # change to True if you want the plots
if plotCDF:
    for k in range(Nf):
        plt.figure()
        a=xx[xx.columns[k]].dropna()
        M=a.shape[0]
        plt.plot(np.sort(a),np.arange(M)/M,label='original dataset')
        plt.plot(np.sort(x_new[x_new.columns[k]]),np.arange(L)/L,label='regressed dataset')
        plt.title('CDF of '+xx.columns[k])
        plt.xlabel('x')
        plt.ylabel('P(X<=x)')
        plt.grid()
        plt.legend(loc='upper left')

#%%------------------ Decision trees -------------------
# Let us use only the complete data (no missing values) to train the decision tree
target = Xtrain.classk
inform = Xtrain.drop('classk', axis=1)
clfXtrain = tree.DecisionTreeClassifier(criterion='entropy',random_state=4)
clfXtrain = clfXtrain.fit(inform,target)
# now we use x_new for the prediction 
test_pred = clfXtrain.predict(x_new.drop('classk', axis=1))
plt.figure(figsize=(10,15))
tree.plot_tree(clfXtrain,feature_names=feat_names[:24],
               class_names=target_names,rounded=True,
               proportion =False ,filled = True )
plt.savefig('fig_training.png' )
#%%
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
print('Performance of the decision tree based on the training dataset only:')
print('accuracy =', accuracy_score(x_new.classk,test_pred))
print(confusion_matrix(x_new.classk,test_pred))



