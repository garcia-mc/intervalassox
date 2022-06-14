from residual_func import lagakos


from sklearn.preprocessing import StandardScaler
from nonlin_cox_gen import nonlin_cox_gen, strong_linear
import matplotlib.pyplot as plt

import torch
import numpy as np

from interfaces import LassoNetRegressor

B=50

for l in range(B):


    z,m,betast=strong_linear()
    zt=torch.from_numpy(z)
    
    
    data=nonlin_cox_gen(z,m) # 999 means infinity
    
    y=data.round(5)
    
    
    X=zt
    
    _, true_features = X.shape
    
    
    # standardize
    #X = StandardScaler().fit_transform(X)
    
    lambda_seq=np.linspace(0.5,2.5,30)
    
    try:
        # =============================================================================
        model = LassoNetRegressor(
            hidden_dims=(10,10),
            eps_start=0.1,
            n_iters=(500, 500), # implement early stopping with 4 nines 
            verbose=True,
            final_lambda=0,
            lambda_seq=lambda_seq,
            M=0,
            final_run=False
        )
        try:
        	path1 = model.path(X.numpy(), y) #or .numpy
        except:
        	pass
        
        pathplot=path1
        
        vlosses=[pathplot[k].val_loss for k in range(len(path1))]
        losses=[pathplot[k].loss for k in range(len(path1))]
        selected=[np.array(pathplot[k].selected) for k in range(len(path1))]
        
        lambdas=[pathplot[k].lambda_ for k in range(len(path1))]
        
        
        
        plt.scatter(lambdas[1:],vlosses[1:],c='red')
        
        plt.savefig("results/validation losses.png", dpi=300)
        plt.clf()
            
        plt.scatter(lambdas,losses,c='blue')
        
        plt.savefig("results/train losses.png", dpi=300)
        plt.clf()
            
        
        #indexes=(np.concatenate((np.array(np.argsort(vlosses)[:2],int),np.array(np.argsort(losses)[:2],int))))
        index=np.array(np.argsort(vlosses[1:])[0]+1,int)
        
        #LAMBDA=np.median(np.array(lambdas)[indexes.astype(int)])
        LAMBDA=np.array(lambdas)[index.astype(int)]
        
        BEST_MODEL=pathplot[index].model_i
        
        TNs=[]
        truth=np.concatenate([[True]*3, [False] * 7])
        for i in range(len(selected)):
            proba=selected[i]
        
        
            TN=np.sum(np.logical_not(np.logical_or(proba,truth)))/7
            TNs.append(TN)
        
        TPs=[]
        for i in range(len(selected)):
            proba=selected[i]
        
        
            TP=np.sum(np.logical_and(proba,truth))/3
            TPs.append(TP)
        
        fig, axs = plt.subplots(2)
        fig.suptitle('results/Vertically stacked subplots')
        
        
        axs[0].scatter(lambdas[1:],vlosses[1:],c='blue')
        
        axs[1].scatter(lambdas[1:],TNs[1:])
        axs[1].scatter(lambdas[1:],TPs[1:],marker='x')
        
        
        
        plt.savefig("results/everything.png", dpi=300)
        plt.clf()
            
        
        znew,mnew,betast=strong_linear()
        Xnew=torch.from_numpy(znew)
        cox=np.zeros(len(mnew))
        for i in range(len(mnew)):
            cov=torch.tensor(Xnew[i,:],dtype=torch.float)
        
            cox[i]=BEST_MODEL.forward(cov).detach().numpy()
            
        plt.scatter(cox,mnew,c='blue')
        plt.savefig("results/fitted vs true.png", dpi=300)
        plt.clf()
            
        
        import scipy
        
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(cox, mnew)
        
        np.savetxt('results/R.txt',[r_value])
        
        
        np.savetxt('results/tps.txt',TPs)
        np.savetxt('results/tns.txt',TNs)
        
        Jinford=model.mnonpar_train.JJ
        gamma=5
        lam=1
        H0=np.asarray((lam/gamma)*np.asmatrix(np.exp(gamma*Jinford[0,:])-1).transpose())
        predi0=torch.zeros(len(Jinford[6,:]),X.shape[1])
        q0=np.multiply(np.asmatrix(Jinford[6,:]).transpose(),np.exp(BEST_MODEL.forward(predi0).detach().numpy()))
        plt.plot(Jinford[0,:], np.asarray(q0), drawstyle='steps-post', label='steps-post',linewidth=0.5,c='blue')
        plt.plot(Jinford[0,:], H0,c='grey')
        
        plt.savefig("results/baselinehazard.png", dpi=300)
        plt.clf()
        file_object = open('rs.txt', 'a')
        # Append 'hello' at the end of file
        file_object.write(str(r_value))
        file_object.write(' ')
        
        file_object = open('results/alltps.txt', 'a')
        # Append 'hello' at the end of file
        file_object.write(str(TPs[index]))
        file_object.write(' ')
        
        file_object = open('results/alltns.txt', 'a')
        # Append 'hello' at the end of file
        file_object.write(str(TNs[index]))
        file_object.write(' ')
    
        # Close the file
        file_object.close()
        
        LAM=np.asarray(q0)
        
        d2=(H0-LAM)**2
        
        distance=np.dot(np.diff(Jinford[0,:]),d2[:-1])
        
        file_object = open('results/distances.txt', 'a')
        # Append 'hello' at the end of file
        file_object.write(str(distance))
        file_object.write(' ')
        
    except:
        pass
    
    
    
    
