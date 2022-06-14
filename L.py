#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 19 09:42:07 2022

@author: carlos
"""
#import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.interpolate import interp1d
import math

###

from rpy2 import robjects

from rpy2.robjects.packages import importr

from statsmodels.distributions.empirical_distribution import ECDF

# imports the base module for R.
base = importr("base")
 
# imports the utils package for R.
utils = importr("utils")

# import fdrtool

utils = importr("fdrtool")

gcmpy = robjects.r['gcmlcm']





robjects.r('''
f <- function(G,V) {
    # library(fdrtool)

    x=c(0,G)
    y=c(0,V)
    
    # plot(x,y)
    
    gcm=gcmlcm(x, y, type=c("gcm"))
    # plot(x,y)
    # points(gcm$x.knots,gcm$y.knots,col='red')
    
    derivatives=gcm$slope.knots
    
    positions=which(x %in% gcm$x.knots) # position of the knots 
    
    newLambda=numeric(length(x))
    k=1
    
    slopes=derivatives
    for(i in 2:length(positions)) {
      while(k<=positions[i]) {
        newLambda[k]=slopes[i-1]
        k=k+1
      }
    }
        
    final=newLambda[-1]
    return(final)
}

''')

###


class L:
  def __init__(self, indata,cov):
    
    
    self.delta1=indata[0];
    self.delta2=indata[1];
    self.delta3=indata[2];
    self.JJ=indata[3];
    
    self.z=cov;
    
    

    self.m=int(self.JJ.shape[1])
    self.eps1=1e-5
    ecdf0=ECDF(self.JJ[0,:])
    gamma=0.1
    lam=0.5
    H0=np.asarray((lam/gamma)*np.asmatrix(np.exp(gamma*self.JJ[0,:])-1))
    #print(ecdf0(self.JJ[0,:]))
    self.Lambda0=-np.log(1-ecdf0(self.JJ[0,:])*0.9)#H0#np.log(1/(1-np.cumsum(np.ones(self.m)/(self.m+1))))

    Zimportant=self.z[self.JJ[1,:].astype(int)].transpose()

    self.JJ=np.vstack((self.JJ,self.Lambda0,Zimportant))
    self.n=len(self.z)
    
  def etz(self,theta,iobs):
        # print(theta)
        return np.exp(theta[int(iobs)])
  def necox(self,theta,Lambda,iobs): # Negative Exponential of Cox model
        
        return torch.exp(-(torch.from_numpy(np.asarray(Lambda))*self.etz(theta,iobs)))
  def a1(self,theta,Lambda,iobs):
        denom=(1-self.necox(theta,Lambda,iobs))
        if abs(denom)<self.eps1:
            denom=self.eps1
        return self.etz(theta,iobs)*self.necox(theta,Lambda,iobs)/denom
      
  def a2(self,theta,Lambdau,Lambdav,iobs):
        denom=(self.necox(theta,Lambdau,iobs)-self.necox(theta,Lambdav,iobs))
        if abs(denom)<self.eps1:
            denom=self.eps1
        return self.etz(theta,iobs)*self.necox(theta,Lambdau,iobs)/denom
  
  def a3(self,theta,Lambdau,Lambdav,iobs):
        denom=(self.necox(theta,Lambdau,iobs)-self.necox(theta,Lambdav,iobs))
        if abs(denom)<self.eps1:
            denom=self.eps1
        return self.etz(theta,iobs)*self.necox(theta,Lambdav ,iobs)/denom
      
  
  def fit(self,theta,nitergr): 
      print('fitting loss nova')
      
      Jinford=self.JJ
      oldcumul=Jinford[6,:]
      for i in range(nitergr):
        eps2=1e-5
        Gjumps=np.zeros(self.m)
        Wjumps=np.zeros(self.m)
        
        # row 1 is patient id
        # row 2 is border type
        # row 3 is delta1
        # row 4 is delta2
        # row 5 is delta3
        # row 6 is Lambda
        # from row 7 on is zeta
        
        for k in range(self.m):
            
            if Jinford[2,k] == 2:
                if Jinford[3,k] !=0:
                    Gjumps[k]=Gjumps[k] + self.a1(theta,Jinford[6,k],Jinford[1,k])**2
                if Jinford[4,k] !=0:
                    Gjumps[k]=Gjumps[k] + self.a2(theta,Jinford[6,k],
                                                   Jinford[6,np.logical_and(Jinford[1,:]==Jinford[1,k],Jinford[2,:]==3)],Jinford[1,k])**2
            if Jinford[2,k] == 3:
                
                if Jinford[4,k] !=0:
                    Gjumps[k]=Gjumps[k] + self.a3(theta,
                                                   Jinford[6,np.logical_and(Jinford[1,:]==Jinford[1,k],Jinford[2,:]==2)],Jinford[6,k],Jinford[1,k])**2
                
                if Jinford[5,k] !=0:
                    Gjumps[k]=Gjumps[k] + self.etz(theta,Jinford[1,k])**2
        
            if np.isnan(Gjumps[k]):
                # print(Gjumps[k])
                print('PROBLEM G')
                Gjumps[k]=np.min(Gjumps[~k])
        for k in range(self.m):
            
            if Jinford[2,k] == 2:
                if Jinford[3,k] !=0:
                    Wjumps[k]=Wjumps[k] + self.a1(theta,Jinford[6,k],Jinford[1,k])
                if Jinford[4,k] !=0:
                    Wjumps[k]=Wjumps[k] - self.a2(theta,Jinford[6,k],
                                                   Jinford[6,np.logical_and(Jinford[1,:]==Jinford[1,k],Jinford[2,:]==3)],Jinford[1,k])
            if Jinford[2,k] == 3:
                
                if Jinford[4,k] !=0:
                    Wjumps[k]=Wjumps[k] + self.a3(theta,
                                                   Jinford[6,np.logical_and(Jinford[1,:]==Jinford[1,k],Jinford[2,:]==2)],Jinford[6,k],Jinford[1,k])
                
                if Jinford[5,k] !=0:
                    Wjumps[k]=Wjumps[k] - self.etz(theta,Jinford[1,k])
            
            if np.isnan(Wjumps[k]):
                    # print(Gjumps[k])
                print('PROBLEM W')
                #print(theta[int(Jinford[1,k])])
                #print(k)
                Wjumps[k]=0                    
                
        
        
        #ts=J
        
        
        # ts=np.linspace(4, 9, num=500)
        #Gvalues=np.zeros(m)
        Gjumps=Gjumps/self.m
        Wjumps=Wjumps/self.m
        W=np.zeros(self.m)
        G=np.zeros(self.m)
        integral=np.zeros(self.m)
        
        G=np.cumsum(Gjumps)
        W=np.cumsum(Wjumps)
        
        integral=np.cumsum(Jinford[6,:]*Gjumps)
        
        
        V=W+integral
        
        # plt.plot(G, V)
        # plt.show()
        #plt.scatter(G,V ,s=0.1)
        #plt.show()
        self.G=G
        self.V=V
        self.Gj=Gjumps
        #print(G)
        #print(V)
        Gr=np.unique(G,return_index=True)[1]

     
       # print(len(Gr))
        #print(self.m)
        #print('PROBLEM R')
        #print(Wjumps)
        #print(Jinford[6,:])
        #print(integral)
        gcmshort=robjects.globalenv['f'](robjects.FloatVector(G[Gr]),robjects.FloatVector(V[Gr]))
        func = interp1d(Jinford[0,Gr], gcmshort,fill_value="extrapolate")
        gcm=func(Jinford[0,:])
        pass
      
        
        Jinford[6,:]=gcm

      self.JJ=Jinford    

  def loss_wellner(self,m):
     ln=torch.zeros(self.n)

     for i in range(self.n):
        if self.delta1[i]==1:
            ln[i]=torch.log(1-torch.exp(-torch.from_numpy(self.JJ[6,np.logical_and(self.JJ[1,:]==i,self.JJ[2,:]==2)])*torch.exp(m[i])))
        if self.delta2[i]==1:
            ln[i]=torch.log(torch.exp(-torch.from_numpy(self.JJ[6,np.logical_and(self.JJ[1,:]==i,self.JJ[2,:]==2)])*torch.exp(m[i]))-torch.exp(-torch.from_numpy(self.JJ[6,np.logical_and(self.JJ[1,:]==i,self.JJ[2,:]==3)])*torch.exp(m[i])))
        if self.delta3[i]==1:
            ln[i]=-torch.from_numpy(self.JJ[6,np.logical_and(self.JJ[1,:]==i,self.JJ[2,:]==3)])*torch.exp(m[i])
     log_likelihood=torch.nanmean(ln)
     return(-log_likelihood)
  def loss_val(self,m,y):
    n=len(y)
    residuals=torch.zeros(n)
    Jinford=self.JJ
    times=Jinford[0,:]
    chs=Jinford[6,:]
    
    chfunc = interp1d(times, chs,fill_value="extrapolate")
    for i in range(n):
        obs=y[i,:]

        cox=np.exp(m[i])
        if obs[3]==1:
        	residual=np.log(np.exp(-chfunc(obs[0])*cox)-np.exp(-chfunc(obs[1])*cox))
            #surva=np.exp(-chfunc(obs[0])*cox)
            #survb=np.exp(-chfunc(obs[1])*cox)
            #residual=(surva*np.log(surva)-survb*np.log(survb))/(surva-survb)

        if obs[2]==1:
        	residual=np.log(1-np.exp(-chfunc(obs[0])*cox))
            #surva=np.exp(-chfunc(obs[0])*cox)
            #survb=0
            #residual=(surva*np.log(surva)-0)/(surva-survb)
        if obs[4]==1:
        	residual=-chfunc(obs[1])*cox
            #survb=np.exp(-chfunc(obs[1])*cox)
            #residual=(surva*np.log(surva)-survb*np.log(survb))/(surva-survb)
        residuals[i]=residual
        

        
            
    #print(residuals)  
    return(-torch.nanmean(residuals[~torch.isneginf(residuals)]))


      