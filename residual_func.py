#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  7 19:46:41 2022

@author: carlos
"""
from scipy.interpolate import interp1d
import numpy as np
import torch 

def lagakos(X,y,model):
    n=len(y)
    residuals=[]
    Jinford=model.mnonpar_full.JJ
    times=Jinford[0,:]
    chs=Jinford[6,:]
    
    chfunc = interp1d(times, chs,fill_value="extrapolate")
    for i in range(n):
        obs=y[i,:]
        cov=torch.tensor(X[i,:],dtype=torch.float)

        cox=np.exp(model.model.forward(cov).detach().numpy())
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
            
        if (~np.isnan(residual) and ~np.isinf(residual) ):
            residuals.append(residual)

        
            
            
    return(-np.mean(residuals))

