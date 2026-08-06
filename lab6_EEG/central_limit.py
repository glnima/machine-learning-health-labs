#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# generate samples drawn from a Gaussian random variable using the two techniuqes:
#    Box-Muller (exact)
#    Central limit theorem (approximate)
# Check the Gaussianity of the obtained vector by
# 1. plotting the normalized histogram
# 2. using the Normal Probability plot (a version of the Q-Q plot)  
# 3. checking Kurtosis and finding the p-value
# 4. using Anderson-Darling test and finding the p-value


import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.special import erfc, erfcinv




#%%
np.random.seed(30)  # set the seed to reproduce the experiment
plt.close('all')
Ns=100
mu = 5 # desired mean
sigma = 2 # desired standard deviation
#case = "central-limit" # either "central-limit" or "Box-Muller"
case = "Box-Muller" # either "central-limit" or "Box-Muller"
if case=="central-limit":
    N=5 # number of independent uniform random variables
    K = sigma*np.sqrt(3)
    xUnif=(np.random.rand(Ns,N)-0.5)*2*K # 2D array, uniformly distributed in -K,K (variance = sigma^2)    
    # generate x as approximately Gaussian:
    x=np.sum(xUnif,axis=1)/np.sqrt(N)+mu# apply formula of central limit and add the desired mean
    plt.figure()# plot the normalized histograms and compare with the theoretical pdfs
    #plt.title("histograms vs pdfs")

    plt.subplot(2,1,1)
    Wbin = 3.5*sigma/(Ns**(1/3))# Scott's Rule to get the bin width
    # check the uniform R.V.:
    tocheck = xUnif[:,0]# just the first column
    Nbins = np.ceil((np.max(tocheck)-np.min(tocheck))/Wbin).astype(int)# number of bins
    u=plt.hist(tocheck,bins=Nbins,density=True,label='generated samples')
    pdf=stats.uniform.pdf(x = u[1],loc=-K, scale=2*K)#Compute the theoretical uniform pdf
    plt.plot(u[1],pdf,label='theoretical uniform')
    plt.xlabel("u")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.subplot(2,1,2)
    # check the obtained Gaussian R.V.:
    Nbins = np.ceil((np.max(x)-np.min(x))/Wbin).astype(int)  
    a=plt.hist(x,bins=Nbins,density=True,label='sum of '+str(N)+' uniform random variables')
    pdf = stats.norm.pdf(x = a[1], loc=mu, scale=sigma) #Compute the theoretical Gaussian pdf
    plt.plot(a[1],pdf,label='theoretical Gaussian')
    plt.xlabel("u")
    plt.grid()
    plt.legend()
    plt.tight_layout()
if case == "Box-Muller":
    xUnif=np.random.rand(Ns,2) # uniformly distributed in [0,1]
    sigma2 = sigma**2 # desired variance
    r = np.sqrt(-2*sigma2*np.log(1-xUnif[:,0]))
    theta = xUnif[:,1]*2*np.pi
    x=r*np.cos(theta)+mu
    plt.figure()
    a=plt.hist(x,bins=100,density=True,label='generated samples')
    pdf = stats.norm.pdf(x = a[1], loc=mu, scale=sigma) #Compute the theoretical Gaussian pdf
    plt.plot(a[1],pdf,label='theoretical Gaussian')
    plt.legend()   
    plt.grid()
   

#%% measured versus theoretical CDF
xs = np.sort(x)
y = np.arange(Ns)/Ns
plt.figure()
plt.plot(xs,y,label='measured')
plt.plot(xs,1-0.5*erfc((xs-mu)/(np.sqrt(2)*sigma)),label='theoretical')
plt.xlabel("u")
plt.ylabel("P(x <= u)")
plt.title("Cumulative Distribution Function (CDF)")
plt.legend()
plt.grid()
#%% Normal probability plot
n = 20 # number of quantiles to plot
# n1 = np.floor(np.sqrt(Ns)).astype(int)
# n = np.min([n,n1])
qs = (2*np.arange(n)+1)/(2*n)
ii = np.floor(qs*Ns).astype(int)
xqs = (xs[ii]+xs[ii+1])/2
#xqs_theory = np.sqrt(2)*sigma*erfcinv(2*(1-qs))+mu
xqs_theory = np.sqrt(2)*erfcinv(2*(1-qs))
plt.figure()
plt.plot(xqs_theory,xqs,'-o',markersize=4,label='measured')
plt.plot(xqs_theory,xqs_theory*sigma+mu,'r',linewidth=2,label='theory')
plt.xlabel('normalized x_q (theory)')
plt.ylabel('x_q (meas)')
plt.title('Normal probability plot')
plt.legend()
plt.grid()
#%% t-score
def tsco(x,mu):
    Ns = len(x)
    m = np.mean(x)
    s2 = np.sum((x-m)**2)/(Ns-1)
    s=np.sqrt(s2)
    tt = (m-mu)/s*np.sqrt(Ns)
    return tt
tx = tsco(x,mu)
print('absolute value of t-score for x is ',np.abs(tx))


#%% excess kurtosis
def exc_kurt(x):
    N=len(x)
    m=np.sum(x)/N
    s4=np.sum((x-m)**4)/N
    s2=np.sum((x-m)**2)/N
    k0=s4/s2**2
    A = (N-1)/(N-2)/(N-3)*((N+1)*k0-3*(N-1));
    return A
ku = exc_kurt(x)
print('absolute vlue of excess kurtosis for x is ',np.abs(ku))
#%% Anderson-Darling test
def A_D(x,mu,sigma):
# -N -\sum_{i=1}^{N}\frac{2i-1}{N} \left[ \ln
# F(y_i)+\ln(1-F(y_{N+1-i}))\right]
    N = len(x)
    xsd = -np.sort(-x)
    Fd = 1-0.5*erfc((xsd-mu)/np.sqrt(2)/sigma);
    Fu = np.flipud(Fd);
    ii=np.arange(1,N+1)
    a=(2*ii-1)/N;
    out = -N-a@(np.log(Fu)+np.log(1-Fd));
    return out
a2 = A_D(x,mu,sigma)
print('Anderson-Darling test a^2 for x is ',a2)
#%% estimate p-values for the chosen statistics using simulation
Nexp = 10000#number of experiments to generate the p-value curve
valT = np.zeros((Nexp,))
valKurt = np.zeros((Nexp,))
valA_D = np.zeros((Nexp,))
for k in range(Nexp):
    xg = np.random.randn(Ns)*sigma + mu # hypothesis H_0 is satisfied
    valT[k] = tsco(xg,mu)# find t-score for vector xg that satisfies hypothesis H_0
    valKurt[k] = exc_kurt(xg) # find kurtosis for vector xg that satisfies hypothesis H_0
    valA_D[k] = A_D(xg,mu,sigma)# find A_D for vector xg that satisfies hypothesis H_0
#%% final plots to check Gaussianity from p-values (two-tail p-value)
#%% t-score
plt.figure()
v1 = np.min(np.abs(valT))
v2 = np.max(np.abs(valT))
vals = np.sort(np.abs(valT))
from scipy.stats import t # t-student distribution imported from scipy.stats
plt.plot(vals,2*(1-t.cdf(vals, Ns-1)),'b',label='theoretical p-value')
plt.plot(vals,1-np.arange(Nexp)/Nexp,'b.',label = 'measured p-value')
plt.semilogy([np.abs(tx),np.abs(tx)],[1e-5,1],'r--',label = 'measured t-score')
plt.semilogy([v1,v2],[0.05,0.05],'k--',label = 'significance level alpha')
plt.grid()
plt.xlabel('x')
plt.ylabel('P(|X|>x)')
plt.legend()
plt.title('t-score')
#%% excess kurtosis
plt.figure()
v1 = np.min(np.abs(valKurt))
v2 = np.max(np.abs(valKurt))
plt.semilogy(np.sort(np.abs(valKurt)),1-np.arange(Nexp)/Nexp,'b',label = 'measured p-value')
plt.semilogy([np.abs(ku),np.abs(ku)],[1e-5,1],'r--',label = 'measured excess kurtosis')
plt.semilogy([v1,v2],[0.05,0.05],'k--',label = 'significance level alpha')
plt.grid()
plt.xlabel('x')
plt.ylabel('P(|X|>x)')
plt.legend()
plt.title('Excess kurtosis')
#%% Anderson-Darling test
plt.figure()
v1 = np.min(np.abs(valA_D))
v2 = np.max(np.abs(valA_D))
plt.semilogy(np.sort(np.abs(valA_D)),1-np.arange(Nexp)/Nexp,'b',label = 'measured p-value')
plt.semilogy([a2,a2],[1e-5,1],'r--',label = 'measured A-D metric')
plt.semilogy([v1,v2],[0.05,0.05],'k--',label = 'significance level alpha')
plt.grid()
plt.xlabel('x')
plt.ylabel('P(|X|>x)')
plt.legend()
plt.title('Anderson-Darling test')

    
