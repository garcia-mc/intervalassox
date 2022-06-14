#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 26 13:01:34 2022

@author: carlos
"""
import numpy as np
from scipy.stats import multivariate_normal as mvn


def strong_linear():
    p = 5
    n = 1000
    coef = np.concatenate([[-0.1,0.5,0.3] ,[0] * (7)])
    print(coef)
    rho=0.5
    cov=np.fromfunction(lambda i, j: rho**np.abs(i-j), (2*p,2*p))

    X = mvn.rvs(mean = [0] * (2*p), cov=cov, size = n)

    linear = X.dot(coef)
    # noise = np.random.randn(n)
    x1, x2, x3, *_ = X.T
    nonlinear = 2 * (x1 ** 3 - 3 * x1) + 4 * (x2 ** 2 * x3 - x3)
    y =  linear  #+ nonlinear/5 #8 * noise
    return X, y, coef

def nonlin_cox_gen(covariates,m):

    n=len(covariates) # number of patients
    
    
    k=9 # number of checkups
    q=0.9 # probability of attendanding next checkup
    
    
    gamma=5# initially both 0.1
    lam=1 
    
    u=np.zeros(n)
    v=np.zeros(n)
    d1=np.zeros(n)
    d2=np.zeros(n)
    d3=np.zeros(n)
    
    
    # generation of latent times
    ut=np.random.uniform(low=0.0, high=1.0, size=n)
    t=(1/gamma)*np.log(1-gamma*np.log(ut)*np.exp(-m)/lam)
    
    
    
    for i in range(n):
        checkups=np.sort(np.random.uniform(low=np.quantile(t,0.2), 
                                       high=np.quantile(t,0.8), size=k))
    
        uk=np.random.uniform(low=0.0, high=1.0, size=k-1)
    
        # first checkup is always attended
        attend=np.insert(uk<q, 0, True, axis=0)
        when=t[i]>checkups
        
        indsu=attend & when
        indsv=attend & (~ when)
        
        if(np.sum(indsu)!=0):
            u[i]=checkups[np.max(np.where(indsu))]
        else:
            d1[i]=1 # datum is left-censored
            u[i]=0
        
        if(np.sum(indsv)!=0):
            v[i]=checkups[np.min(np.where(indsv))]
        else:
            d3[i]=1 # datum is right-censored
            v[i]=999
        d2[i]=1-d1[i]-d3[i]

        if(d2[i]==0): 
            u[i],v[i]=v[i],u[i]
        
            
    np.column_stack((u,v,d1,d2,d3))
    return(np.column_stack((u,v,d1,d2,d3,t)))
