import numpy as np
import scipy.io as scio
import torch 
from pyDOE import *
import random
from scipy.stats import qmc
import math
#from pyswarm import pso

from scipy.interpolate import griddata
from scipy.optimize import minimize
from scipy.optimize import shgo
import time

from . import GP_processing
torch.set_default_dtype(torch.double)

class Generating_data(object):
    def __init__(self, para_theta=None, pde_operator=1, source_term= 1, sigma_e=0.001, noisy=True, boundary_condition = False, noisy_known = False, design_instance = None, n_obs =90, n_I = 90, theta_instance = None):
        self.pde_operator=pde_operator
        self.source_term=source_term
        self.para_theta=para_theta # identify initial gauss for parameter
        self.noisy_known=noisy_known
        self.sigma_e_prop=sigma_e
        self.n_obs = n_obs
        self.n_I = n_I
        self.theta_instance = theta_instance
        self.boundary_condition = boundary_condition
        self.design_instance = design_instance
        if self.n_I < self.n_obs : self.n_I = self.n_obs
        self._Load_Data()
        self._GP_Preprocess(noisy=noisy, noisy_known=noisy_known, pde_operator = pde_operator)

    def _Load_Data(self):
        if self.source_term==1:
            # the operator is \partial/\partial x_1 +\partial/\partial x_2            
            self.x_range = torch.tensor([[0. ,0.],[1. ,1.]])
            self.Generating_Design(2, self.n_obs, self.n_I, x_obs = None, x_range = None)
            self.x_pred=torch.tensor([np.arange(60) / 59,np.arange(60) / 59]).T
            self.y_pred=self.True_Solution(self.x_pred)
            self.n_obs, self.p = self.y_obs.size()
            self.n_I, self.d = self.x_I.size()
            self.x_bound=torch.zeros(0,self.d)
            self.y_bound=torch.zeros(0,self.p)
            if self.boundary_condition is False :
                self.x_bound=torch.zeros(0,self.d)
                self.y_bound=torch.zeros(0,self.p)
            self.n_bound = self.y_bound.shape[0]
            self.x_all=torch.cat((self.x_I,self.x_bound), 0)
            self.y_all=torch.cat((self.y_sol,self.y_bound), 0)
            p=self.theta_true.shape
            if self.para_theta is None :
                self.para_theta=torch.rand(p).double()
        elif self.source_term==0:
            # the operator is \partial^2/\partial x_1^2 +\partial^2/\partial x_2^2
            self.x_range = torch.tensor([[0., 0.],[1. ,1.]])
            self.x_pred=torch.tensor([np.arange(60) / 59,np.arange(60) / 59]).T
            self.y_pred=self.True_Solution(self.x_pred)
            self.Generating_Design(2, self.n_obs, self.n_I, x_obs = None, x_range = None)
            self.n_obs, self.p = self.y_obs.size()
            self.n_I, self.d = self.x_I.size()
            self.x_bound=torch.zeros(0,self.d)
            self.y_bound=torch.zeros(0,self.p)
            if self.boundary_condition is False :
                self.x_bound=torch.zeros(0,self.d)
                self.y_bound=torch.zeros(0,self.p)
            self.n_bound = self.y_bound.shape[0]
            self.x_all=torch.cat((self.x_I,self.x_bound), 0)
            self.y_all=torch.cat((self.y_sol,self.y_bound), 0)
            p=self.theta_true.shape
            if self.para_theta is None :
                self.para_theta=torch.rand(p).double()
        elif self.source_term==2:
            # the operator is \partial^2/\partial x_1^2 +\partial^2/\partial x_2^2
            self.x_range = torch.tensor([[0., 0.],[1., 1.]])
            path = 'data/eg_conductivity.mat'
            data = scio.loadmat(path)
            self.true_data = data
            self.Generating_Design(2, self.n_obs, self.n_I, x_obs = None, x_range = None)
            self.x_pred=torch.tensor(data['x_pred']).double()
            self.y_pred=torch.tensor(data['y_pred']).T
            self.n_obs, self.p = self.y_obs.size()
            self.n_I, self.d = self.x_I.size()
            self.x_bound=torch.tensor(data['x_bound']).double()
            self.y_bound=torch.tensor(data['y_bound']).T
            if self.boundary_condition is False :
                self.x_bound=torch.zeros(0,self.d)
                self.y_bound=torch.zeros(0,self.p)
            self.n_bound = self.y_bound.shape[0]
            self.x_all=torch.cat((self.x_I,self.x_bound), 0)
            self.y_all=torch.cat((self.y_sol,self.y_bound), 0)
            self.theta_true=torch.tensor(data['theta_true'])
            self.theta_true=self.theta_true.reshape(1)
            p=self.theta_true.shape
            if self.para_theta is None :
                self.para_theta=torch.rand(p).double()
        elif self.source_term==3:
            # the operator is \partial/\partial t - \partial^2/\partial x_1^2 - \partial^2/\partial x_2^2
            self.x_range = torch.tensor([[0.,0.,0.],[0.15, 1., 1.]])
            path = 'data/eg_contaminant.mat'
            data = scio.loadmat(path)
            self.true_data = data

            self.x_obs = torch.tensor(data['x_obs']).double()
            self.x_I = torch.tensor(data['x_obs']).double()
            self.y_sol = torch.tensor(data['y_obs']).T
            self.y_obs = torch.tensor(data['y_obs']).T
            self.aIdx = torch.tensor(range(self.x_I.shape[0]))

            self.Generating_Design(3, self.n_obs, self.n_I, x_obs = None, x_range = None)
            self.x_pred=torch.tensor(data['x_pred']).double()
            self.y_pred=torch.tensor(data['y_pred']).T
            self.x_bound=torch.tensor(data['x_bound']).double()
            self.y_bound=torch.tensor(data['y_bound']).T
            self.theta_true=torch.tensor(data['theta_true']).squeeze()
            self.n_obs, self.p = self.y_obs.size()
            self.n_I, self.d = self.x_I.size()
            if self.boundary_condition is False :
                self.x_bound=torch.zeros(0,self.d)
                self.y_bound=torch.zeros(0,self.p)
            self.n_bound = self.y_bound.shape[0]
            self.x_all=torch.cat((self.x_I,self.x_bound), 0)
            self.y_all=torch.cat((self.y_sol,self.y_bound), 0)
            p=self.theta_true.shape
            if self.para_theta is None :
                self.para_theta=torch.rand(p).double()
        elif self.source_term==4:
            # the operator is \partial^2/\partial x_1^2 +\partial^2/\partial x_2^2
            self.x_range = torch.tensor([[0. ,0. ],[20., 40.]])
            path = 'data/eg_ladar.mat'
            data = scio.loadmat(path)
            self.true_data = data
            self.Generating_Design(2, self.n_obs, self.n_I, x_obs = None, x_range = None)
            self.x_pred=torch.tensor(data['x_pred'])
            self.y_pred=torch.tensor(data['y_pred']).T
            n , p= self.y_sol.size()
            self.x_bound=torch.tensor(data['x_bound']).double()
            self.y_bound=torch.tensor(data['y_bound']).T
            self.n_obs, self.p = self.y_obs.size()
            self.n_I, self.d = self.x_I.size()
            #self.x_bound=self.x_bound[9:18,:]
            #self.y_bound=self.y_bound[9:18,:]
            if self.boundary_condition is False :
                self.x_bound=torch.zeros(0,self.d)
                self.y_bound=torch.zeros(0,self.p)
            self.n_bound = self.y_bound.shape[0]
            self.x_all=torch.cat((self.x_I,self.x_bound), 0)
            self.y_all=torch.cat((self.y_sol,self.y_bound), 0)
            self.theta_true=torch.tensor(data['theta_true']).double()
            self.theta_true=self.theta_true.squeeze()
            p=self.theta_true.shape
            if self.para_theta is None :
                self.para_theta=torch.rand(p).double()
        elif self.source_term==5:
            # the operator is \partial^2/\partial x_1^2 +\partial^2/\partial x_2^2
            self.x_range = torch.tensor([[0. ,0. ],[0.1, 1.]])
            if self.theta_instance is None : path = 'data/eg_burger.mat'
            else : path = 'data/eg_burger1.mat'
            data = scio.loadmat(path)
            self.true_data = data
            self.Generating_Design(2, self.n_obs, self.n_I, x_obs = None)
            self.x_pred=torch.tensor(data['x_pred'])
            self.y_pred=torch.tensor(data['y_pred']).T
            n , p= self.y_sol.size()
            self.x_bound=torch.tensor(data['x_bound']).double()
            self.y_bound=torch.tensor(data['y_bound']).T
            self.n_obs, self.p = self.y_obs.size()
            self.n_I, self.d = self.x_I.size()
            if self.boundary_condition is False :
                self.x_bound=torch.zeros(0,self.d)
                self.y_bound=torch.zeros(0,self.p)
            self.n_bound = self.y_bound.shape[0]
            self.x_all=torch.cat((self.x_I,self.x_bound), 0)
            self.y_all=torch.cat((self.y_sol,self.y_bound), 0)
            self.theta_true=torch.tensor(data['theta_true']).double()
            self.theta_true=self.theta_true.squeeze()
            p=self.theta_true.shape
            if self.para_theta is None :
                self.para_theta=torch.rand(p).double()

        elif self.source_term==6 or self.source_term==7 or self.source_term==8 :
            # the operator is  
            self.x_range = torch.tensor([[0. ,0. ,0.],[1.,1.,1.]])
            path = 'data/eg_brusselator.mat'
            data = scio.loadmat(path)
            self.true_data = data
            self.Generating_Design(3, self.n_obs, self.n_I, x_obs = None)
            self.x_pred=torch.tensor(data['x_pred'])
            self.y_pred=torch.tensor(data['y_pred']).T
            n , p= self.y_sol.size()
            self.n_I, self.d = self.x_I.size()
            self.n_obs, self.p = self.y_obs.size()
            self.x_bound=torch.zeros(0,self.d)
            self.y_bound=torch.zeros(0,self.p)
            if self.boundary_condition is False :
                self.x_bound=torch.zeros(0,self.d)
                self.y_bound=torch.zeros(0,self.p)
            self.n_bound = self.y_bound.shape[0]
            self.x_all=torch.cat((self.x_I,self.x_bound), 0)
            self.y_all=torch.cat((self.y_sol,self.y_bound), 0)
            self.theta_true=torch.tensor(data['theta_true']).double()
            self.theta_true=self.theta_true.squeeze()
            p=self.theta_true.shape
            if self.para_theta is None :
                self.para_theta=torch.rand(p).double()




    def _GP_Preprocess(self, noisy, noisy_known, pde_operator):
        '''
        GP preprocessing of the Input Data
        '''
        self.GP_Components = []
        self.GP_PDE_Components = []
        self.GP_Models=[]
        for i in range(self.p):
            # available observation index
            #aIdx = ~torch.isnan(self.y_obs[:,i]) 
            if (pde_operator==1): 
                self.nu=2.01
            elif (pde_operator>=2): 
                self.nu=4.1
            GP_model=GP_processing.GP_modeling(self, noisy = noisy, nu=self.nu, noisy_known=noisy_known)
            GP_model.Train_GP(self, ind_y = i)
            #GP_PDE_component=GP_model.Calculating_Cov_Component()
            self.GP_Components.append({
                #'aIdx':aIdx, # non-missing data index
                'mean':GP_model.mean,
                'kernel':GP_model.kernel,
                'outputscale':GP_model.outputscale,
                'noisescale':GP_model.noisescale,
                'corr_data':GP_model.R
            })
            self.GP_Models.append(GP_model)
            #self.GP_PDE_Components.append(GP_PDE_component)
        
        if self.pde_operator >=4:
            self.x_I = torch.cat((self.x_I,self.x_bound))
            self.x_obs = torch.cat((self.x_obs,self.x_bound))
            self.y_obs = torch.cat((self.y_obs,self.y_bound))
            self.n_bound = self.y_bound.shape[0] 
            self.n_obs = self.x_obs.shape[0]
            self.n_I = self.x_I.shape[0]
            self.aIdx = np.hstack((self.aIdx,np.linspace(self.n_I-self.n_bound, self.n_I-1,self.n_bound)))
            self.x_bound=torch.zeros(0,self.d)
            self.y_bound=torch.zeros(0,self.p)     
                
    def Source(self, x_input, u_sol=None, para_theta =None):
        if (self.source_term==1):
            f =  (para_theta[0] + x_input[:,0]*para_theta[1] + x_input[:,0]**2 * para_theta[2])
            f = f.reshape(-1,1)
            f = f * u_sol
        elif(self.source_term==0): # toy example for source identification
            f =  (para_theta[0]+para_theta[1]*(x_input[:,0]**2)+para_theta[2]*(x_input[:,1]**2))
            f = f.reshape(-1,1)
            f = f*u_sol#(u_sol**para_theta[3])
        elif(self.source_term==2):
            f =  -1 / (2 * math.pi * 0.1**2) *torch.exp(-((x_input[:,0]-1/2)**2+(x_input[:,1]-1/2)**2)/(2*0.1**2))/para_theta
            f = f.reshape(-1,1)
        elif(self.source_term==3):
            if (para_theta is None):
                para_theta=self.para_theta
            f =  para_theta[0] / (2 * math.pi * 0.1**2) *torch.exp(-((x_input[:,1]-para_theta[1])**2+(x_input[:,2]-para_theta[2])**2)/(2*0.1**2))
            f = f.reshape(-1,1)
        elif(self.source_term==4):
            f = torch.zeros(self.n_I,3)
            if (para_theta is None):
                para_theta=self.para_theta
            f[:,0] = u_sol[:,1]
            f[:,1] = u_sol[:,2]
            f[:,2] = para_theta[0]*u_sol[:,1] + para_theta[1]*u_sol[:,2] + para_theta[2]*u_sol[:,0]
        elif(self.source_term==5):
            f = torch.zeros(self.n_I,3)
            if (para_theta is None):
                para_theta=self.para_theta
            f[:,0] = u_sol[:,1]
            f[:,1] = u_sol[:,2]
            f[:,2] = -para_theta[0]*u_sol[:,0]* u_sol[:,1]+ para_theta[1]*u_sol[:,2]
        elif(self.source_term==6 or self.source_term==7 or self.source_term==8 ):
            f = torch.zeros(u_sol.shape[0],4)
            if (para_theta is None):
                para_theta=self.para_theta
            f[:,0] = u_sol[:,2]
            f[:,1] = u_sol[:,3]
            f[:,2] = para_theta[0] * u_sol[:,2] + para_theta[1] - (para_theta[2] + 1) * u_sol[:,0] + u_sol[:,1] * u_sol[:,0]**2
            f[:,3] = para_theta[0] * u_sol[:,3] + para_theta[2]*u_sol[:,0] - u_sol[:,1] * u_sol[:,0]**2
        return(f)
    
    def True_Solution(self, xinput): # for toy examples, where the true function/PDE solution is known analytically. 
        if (self.source_term==1):
            self.theta_true=torch.tensor([1,-2,0])
            if (len(xinput.size())==1): y_=torch.exp(-xinput[0]**2+xinput[1])
            else: 
                y_=torch.exp(-xinput[:,0]**2+xinput[:,1])
                y_=y_.reshape(-1,1)
        elif(self.source_term==0):
            self.theta_true=torch.tensor([-4,4,4])
            if (len(xinput.size())==1):
                y_=torch.exp(-xinput[0]**2-xinput[1]**2)
                #y_=torch.exp(xinput[0]**2+xinput[1]**2)
                #y_=torch.exp(-xinput[0]**2-2*xinput[1])
            else:
                y_=torch.exp(-xinput[:,0]**2-xinput[:,1]**2)
                #y_=torch.exp(xinput[:,0]**2+xinput[:,1]**2)
                #y_=torch.exp(-xinput[:,0]**2-2*xinput[:,1])
                y_=y_.reshape(-1,1)
        elif(self.source_term>=2 and self.source_term<=5):
            data = self.true_data 
            x_inter=torch.tensor(data['x_inter']).double()
            y_inter=torch.tensor(data['y_inter']).T
            y_predict = griddata(x_inter.numpy(), y_inter.numpy(), xinput.numpy(), method='linear')
            y_ = torch.tensor(y_predict)
            if (len(xinput.size()) > 1): y_=y_.reshape(-1,1)
        elif(self.source_term==6 or self.source_term==7 or self.source_term==8 ):
            data = self.true_data 
            x_inter=torch.tensor(data['x_inter']).double()
            y_inter=torch.tensor(data['y_inter']).T
            y_predict = torch.zeros(xinput.shape[0],2)
            for i in range (2):
                y_predict[:,i] = torch.tensor(griddata(x_inter.numpy(), y_inter[:,i].numpy(), xinput.numpy(), method='linear'))
            y_ = y_predict
        return (y_)



    def Generating_Design(self, p, n_obs, n_I, x_obs = None, x_range = None, obs_already = True):
        if x_range is None : x_range = self.x_range
        if x_obs is not None : x_obs = (x_obs - x_range[0,:]) / (x_range[1,:] - x_range[0,:])
        if n_obs > n_I : n_I = n_obs
        sampler = qmc.Halton(d=p, scramble=False)
        if x_obs is None : # randimly generate x_obs and x_I
            if obs_already is True :
                if self.design_instance is not None:
                    if n_obs == 30 or n_obs == 60 or n_obs == 120 or n_obs == 240:
                        x_obs = self.Load_Design(p, n_obs,x_range)
                    else:
                        x_obs = torch.tensor(sampler.random(n=n_obs))
                else:
                    x_obs = torch.tensor(sampler.random(n=n_obs))
                n_add = n_I - n_obs
                x_candidate = torch.tensor(sampler.random(n=15*n_I))
                x_I = self.Combine_design(x_obs, x_candidate, n_add)
                ind_sample = np.arange(n_obs)
            else: 
                sampler = qmc.Halton(d=p, scramble=False)
                x_I = torch.tensor(sampler.random(n=n_I))
                ind_rand=random.sample(range(n_I), n_I) 
                ind_sample=ind_rand[0:n_obs]
                x_obs = x_I[ind_sample,:]
        else: # randomly generate x_I starting from x_obs
            n_add = n_I - n_obs
            sampler = qmc.Halton(d=p, scramble=False)
            x_candidate = torch.tensor(sampler.random(n=15*n_I))
            x_I = self.Combine_design(x_obs, x_candidate, n_add)
            ind_sample = np.arange(n_obs)
        self.x_obs = x_obs * (x_range[1,:] - x_range[0,:]) + x_range[0,:]
        self.x_I = x_I * (x_range[1,:] - x_range[0,:]) + x_range[0,:]
        
        self.y_sol = self.True_Solution(self.x_I)
        self.sigma_e = torch.tensor(self.sigma_e_prop)
        self.sigma_e = self.sigma_e.reshape(1,-1)
        if self.pde_operator ==6 or self.pde_operator==7 or self.pde_operator==8 : self.sigma_e = self.sigma_e * torch.ones(1,2)
        print(self.sigma_e)

        self.y_sol = self.y_sol + self.sigma_e * torch.randn(self.y_sol.shape)
        #print(self.y_sol)
        self.y_obs = self.y_sol [ind_sample,:]
        self.aIdx = ind_sample
    
    def Combine_design(self, x_obs, x_candidate, n_add):
        X = torch.cat((x_obs,x_candidate))
        D = torch.cdist(X, X, p=2) #self.Distance_Design(X,X)[1]
        n_obs = x_obs.shape[0]
        ind = np.array(range(n_obs))
        ind_candidate = np.array(range(x_candidate.shape[0])) + n_obs
        for i in range(n_add):
            D_cut = D [ind,:]
            D_cut = D_cut [:,ind_candidate]
            D_profile = torch.min(D_cut,0).values
            d = torch.argmax(D_profile)
            ind_add = ind_candidate[d]
            ind = np.append(ind,ind_add)
            ind_candidate = np.delete(ind_candidate, d)
        x_obs = X[ind,:]
        return(x_obs)

    def Distance_Design(self, x, D2):
        #D2, = args
        if len(x.shape) ==1:
            n1 = 1
            x = x.reshape[1,-1]
        else: n1 = x.shape[0]
        #print(D2)
        n2 = D2.shape[0]
        Dist = torch.zeros(n1,n2)
        for j in range(n1):
            for i in range(n2):
                Dist[j,i] = torch.mean(torch.square(x[j,:] - D2[i,:]))
        return(-torch.min(Dist),Dist)
        
    def Load_Design(self, p, n_obs, x_range):
        num = self.design_instance
        if n_obs ==30: k = 0
        elif n_obs ==60: k = 1
        elif n_obs ==120: k = 2
        elif n_obs ==240: k = 3
        path = 'data/design_'+str(p)+'d.mat'
        data = scio.loadmat(path)
        name = 'design_seed'
        design = data[name][0][k]
        x_obs = torch.tensor(design[0,num])
        #x_obs = x_obs * (x_range[1,:] - x_range[0,:]) + x_range[0,:]
        #print(x_obs)
        return (x_obs)

    def Update_Prediction_Points(self, source_term = None):
        # return new prediction point set and prediciton for PDE solution
        # for toy example, use analytical solution, for application examples, use interpolation 
        if source_term is None : source_term = self.source_term
        if source_term <= 2 :
            ind = torch.tensor(np.linspace(0,1,30))
            ind = ind. reshape(-1,1)
            self.x_pred = torch.cat((ind, ind),1)
            self.y_pred = self.True_Solution(self.x_pred)
            return (self.x_pred, self.y_pred)
        if source_term == 3 :
            ind = torch.tensor(np.linspace(0,1,30))
            ind = ind. reshape(-1,1)
            self.x_pred = torch.cat((0.1 * torch.ones(30,1), ind, ind),1)
            self.y_pred = self.True_Solution(self.x_pred)
            return (self.x_pred, self.y_pred)
        if source_term == 4 :
            ind = torch.tensor(np.linspace(0,20,30))
            ind = ind. reshape(-1,1)
            self.x_pred = torch.cat((ind, 2*ind),1)
            self.y_pred = self.True_Solution(self.x_pred)
            return (self.x_pred, self.y_pred)
        
        




