import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.cluster as sk
plt.rcParams["font.family"] = "Times New Roman"
#%%
plt.close('all')
xx=pd.read_csv("covid_serological_results.csv")
xx=xx[xx.COVID_swab_res!=1]# remove unclear results
#xx.COVID_swab_res[xx.COVID_swab_res==2]=1# set swab result from 2 to 1 for ill patients
xx.loc[xx.COVID_swab_res==2,"COVID_swab_res"]=1
#%%
swab=xx.COVID_swab_res.values# results from swab: 0= no illness, 1=illness
Test1=xx.IgG_Test1_titre.values
Test2=xx.IgG_Test2_titre.values
#%%
x=Test2
y=swab
x0=x[swab==0] # test results for healthy patients
x1=x[swab==1] # test results for ill patients
Np=np.sum(swab==1) # number of ill patients
Nn=np.sum(swab==0) # number of healthy patients
thresh = 5 # example of threshold
n1=np.sum(x1>thresh) # number of true positives for the given thresh
sens=n1/Np # sensitivity
n0=np.sum(x0<thresh) # number of true negatives
spec=n0/Nn # specificity
print('specificity P(T_n|H) for threshold 5, Test2 =',spec)
print('sensitivity P(T_p|D) for threshold 5, Test2 =',sens)

