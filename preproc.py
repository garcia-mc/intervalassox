#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 17:23:18 2022

@author: duser1
"""

import numpy as np
def preproc(data):
    indata=data
    u=indata[:,0];
    v=indata[:,1];
    delta1=indata[:,2];
    delta2=indata[:,3];
    delta3=indata[:,4];
    
    
    
    changelast=0
    changefirst=0
    
    ###
    count=0
    while(1):
    
        J1n=delta1*u+delta2*u
        J2n=delta3*v+delta2*v
        
        indu=np.squeeze(np.where(J1n >0))
        indv=np.squeeze(np.where(J2n >0))
        ind=np.concatenate((indu,indv),axis=0).astype(int)
        who=np.concatenate((np.repeat(2,len(indu)),np.repeat(3,len(indv))))
        
        # 2 means u 3 means v
        
        J=np.concatenate((J1n[indu],J2n[indv]))
        
        
        jdelta1=np.concatenate((delta1[indu],delta1[indv]))
        jdelta2=np.concatenate((delta2[indu],delta2[indv]))
        jdelta3=np.concatenate((delta3[indu],delta3[indv]))
        
        
        Jinfo=np.stack((J,ind,who,jdelta1,jdelta2,jdelta3))
        
        Jinford=Jinfo[:,np.argsort(J)]
        last=int(Jinford.shape[1])
        if(Jinford[3,0]==1 and Jinford[5,last-1]==1):
            break
        if(count==0):
            Jinford0=Jinford
    
        if(Jinford[3,0]!=1):
                ind=int(Jinford[1,0])
                u[ind]=v[ind]
                v[ind]=0
                delta1[ind]=1
                delta2[ind]=0
                changefirst=changefirst+1
            
            
        
        if(Jinford[5,last-1]!=1):
                ind2=int(Jinford[1,last-1])
                v[ind2]=u[ind2]
                u[ind2]=999
                delta3[ind2]=1
                delta2[ind2]=0
                changelast=changelast+1
        
        count=count+1

    return([delta1,delta2,delta3,Jinford])


