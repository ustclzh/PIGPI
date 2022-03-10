from numpy.core.numeric import Inf
import torch
import time
import numpy as np
import scipy.stats
from pyDOE import *
from scipy.optimize import minimize
from . import GP_processing
from . import HMC
from scipy.optimize import dual_annealing
from scipy.optimize import basinhopping
from scipy.optimize import rosen, shgo
from scipy.optimize import rosen, differential_evolution
torch.set_default_dtype(torch.double)

class MAGI_PDE_Infer(object):
    def __init__(self, True_Model, KL=False):
        self.PDE_Model = True_Model
        self.KL= KL
        self.pde_operator=True_Model.pde_operator
        self.aIdx = True_Model.aIdx
        self.noisy_known=True_Model.noisy_known
        self.x_I=True_Model.x_I
        self.n_I, self.d = self.x_I.size()
        self.n_bound = True_Model.n_bound
        self.y_obs = True_Model.y_obs
        self.n_obs, self.p = self.y_obs.size()
        self.x_obs = True_Model.x_obs
        self.y_bound=True_Model.y_bound
        self.x_bound=True_Model.x_bound
        self.y_all=True_Model.y_all
        self.x_all=True_Model.x_all
        self.GP_Components=True_Model.GP_Components
        self.GP_Models=True_Model.GP_Models
        #self.GP_PDE_Components=True_Model.GP_PDE_Components

    def map(self, nEpoch = 2500):
        u_KL, u, Lu_GP, Lu_GP_KL, GP_Trans_u_Lu = self._Pre_Process()
        time0 = time.time()
        # optimize the initial theta
        para_theta = self.PDE_Model.para_theta
        d_theta = para_theta.shape[0]
        self.d_theta = d_theta

        if self.PDE_Model.source_term <=6:
            u_current = u
            current_opt = np.Inf
            current_theta=para_theta
            theta_cand=lhs(d_theta, samples=d_theta*5, criterion='maximin')
            for ini in range(d_theta*5):
                if self.PDE_Model.pde_operator == 3:
                    bnds=((0,None),(0,1),(0,1))                     
                    res = minimize(self.Loss_Theta_Marginal,(torch.tensor(theta_cand[ini,:])), args=(u_current, Lu_GP_KL, GP_Trans_u_Lu, Lu_GP, self.pde_operator), method='Nelder-Mead', bounds=bnds,  options={'ftol': 1e-6})
                elif self.PDE_Model.pde_operator == 4:
                    bnds=((0,10),(0,10),(0,10))
                    res = minimize(self.Loss_Theta_Marginal,(torch.tensor(theta_cand[ini,:])), args=(u_current, Lu_GP_KL, GP_Trans_u_Lu, Lu_GP, self.pde_operator), method='Nelder-Mead', bounds=bnds,  options={'ftol': 1e-6})
                else:
                    res = minimize(self.Loss_Theta_Marginal,(torch.tensor(theta_cand[ini,:])), args=(u_current, Lu_GP_KL, GP_Trans_u_Lu, Lu_GP, self.pde_operator), method='Nelder-Mead',  options={'ftol': 1e-6})
                if res['fun']<current_opt: 
                    current_theta=res['x']
                    current_opt=res['fun']
            print('Initial estimates for theta:',current_theta, 'True:', self.PDE_Model.theta_true)
            para_theta = torch.tensor(current_theta).double()
            #para_theta = torch.tensor(self.PDE_Model.theta_true).double()
            #para_theta = torch.tensor(torch.ones(current_theta.shape)).double()
            if self.pde_operator>=4:
                para_theta = torch.abs(para_theta)
        elif self.PDE_Model.source_term == 7:
            u_current = u
            current_opt = np.Inf
            current_theta=para_theta
            self.u_KL_initial = u_KL
            lw = [0.001,0.001,0.001] + [-5] * u_KL.shape[0]
            up = [3,3,3] + [5] * u_KL.shape[0]
            bounds=list(zip(lw, up))
            if self.global_alg ==1 :
                res = dual_annealing(self.Loss_for_Censored_Component, bounds=bounds, maxiter=10000)
                x = torch.cat((torch.tensor([0.1,2,1]),u_KL[:,1]))
                print(self.Loss_for_Censored_Component(x.numpy()))
                print(res)
                x = res.x
                self.Loss_for_Censored_Component(x)
            elif self.global_alg ==2:
                res2 = shgo(self.Loss_for_Censored_Component, bounds = bounds, iters=3)
                print(res2)
                x = res.x
            elif self.global_alg ==3:
                minimizer_kwargs = {"method":"L-BFGS-B", "jac":True}
                x0 = torch.cat((para_theta,u_KL[:,1])).numpy()
                res3 = basinhopping(self.Loss_for_Censored_Component, x0, minimizer_kwargs=minimizer_kwargs, niter=20)
                print(res3)
                x = res.x
            elif self.global_alg ==4:
                res4 = differential_evolution(self.Loss_for_Censored_Component, bounds= bounds, updating='deferred',workers=2)
                print(res4)
                x = res.x
                para_theta = x[0:4]
            else: x = torch.randn(4+u_KL.shape[0])
            u_KL_censored = x[4:4+u_KL.shape[0]]
            if self.global_alg <=4 :
                u_KL[:,1] = torch.tensor(u_KL_censored)
                u_3 = self.GP_Trans_u_Lu[1]['Lu_mean_I']  + self.GP_Trans_u_Lu[1]['u_KL_to_Lu_GP'] @ u_KL[:,1]
                u_KL[:,3] = self.GP_Trans_u_Lu[1]['u_to_u_KL'] @ (u_3 - self.GP_Components[3]['mean'])
        # optimize over all parameters
        if self.PDE_Model.source_term == 0: u_lr = 5e-2
        elif self.PDE_Model.source_term == 1: u_lr = 1e-1
        elif self.PDE_Model.source_term == 2: u_lr = 5e-2
        elif self.PDE_Model.source_term == 3: u_lr= 1e-1
        elif self.PDE_Model.source_term == 4: u_lr= 5e-2
        elif self.PDE_Model.source_term == 5: u_lr= 5e-2
        elif self.PDE_Model.source_term == 6: u_lr= 5e-2
        elif self.PDE_Model.source_term == 7: 
            u_lr= 1
            para_theta = torch.rand(d_theta).double()
            #if self.cheat ==1 or self.cheat ==2: para_theta = self.PDE_Model.theta_true.clone()
        elif self.PDE_Model.source_term == 8: 
            u_lr= 0.1
            para_theta = torch.ones(d_theta).double()
        u_KL=u_KL.requires_grad_()
        para_theta=para_theta.requires_grad_()
        
        p = self.y_obs.shape[1]
        lognoisescale = torch.zeros(p)
        for i in range(p):
            lognoisescale[i]=torch.log(self.GP_Components[i]['noisescale'].double())
        if self.noisy_known is False  :
            lognoisescale=lognoisescale.requires_grad_()
            self.optimizer_u_theta = torch.optim.LBFGS([u_KL,para_theta,lognoisescale], lr = u_lr)
        else :
            self.optimizer_u_theta = torch.optim.LBFGS([u_KL,para_theta], lr = u_lr)
            lognoisescale_opt=lognoisescale
        #self.optimizer_u_theta = torch.optim.SGD([u,para_theta], lr = u_lr, momentum=0.8)
        #pointwise_lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer_u_theta, step_size=500, gamma=0.95)
        pointwise_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer_u_theta, lr_lambda = lambda epoch: 1/((epoch+1)**0.5), last_epoch=-1)
        print('start optimiza theta and u:')

        U_KL_trace = ()
        def closure():
            self.optimizer_u_theta.zero_grad()
            loss = self.Minus_Log_Posterior(u_KL, para_theta, lognoisescale / 2)
            loss.backward()
            return loss
        for epoch in range(nEpoch):
            self.optimizer_u_theta.zero_grad()
            loss_u_theta = self.Minus_Log_Posterior(u_KL, para_theta, lognoisescale / 2)
            if epoch==0:
                loss_u_theta_opt=loss_u_theta.clone().detach()
                u_KL_opt=u_KL.clone().detach()
                theta_opt=para_theta.clone().detach()
                if self.noisy_known is False : lognoisescale_opt=lognoisescale.clone().detach()
            else:
                #if para_theta[0]<0: para_theta[0] = torch.abs(para_theta[0])
                if loss_u_theta<loss_u_theta_opt:
                    loss_u_theta_opt=loss_u_theta.clone().detach()
                    u_KL_opt=u_KL.clone().detach()
                    theta_opt=para_theta.clone().detach()
                    if self.noisy_known is False : lognoisescale_opt=lognoisescale.clone().detach()
            loss_u_theta.backward()
            self.optimizer_u_theta.step(closure)
            pointwise_lr_scheduler.step()
            if (np.isnan(self.Minus_Log_Posterior(u_KL, para_theta, lognoisescale / 2).detach().numpy())):
                u_KL = u_KL_opt
                para_theta = theta_opt
                if self.noisy_known is False : lognoisescale=lognoisescale_opt
            if (epoch+1) % 500 == 0 :
                #print(para_theta)
                print(epoch+1, '/', nEpoch, 'current opt: theta:', theta_opt.numpy(),'error/out_scale', torch.exp(lognoisescale_opt).clone().detach().numpy()/self.GP_Components[0]['outputscale'])
                #print('current state: theta:', para_theta.clone().detach().numpy(),'error/out_scale', torch.exp(lognoisescale).clone().detach().numpy()/self.GP_Components[0]['outputscale'])
                print('gradient', torch.mean(torch.abs(u_KL.grad.squeeze())).numpy(), para_theta.grad.numpy())
                #'loss:', loss_u_theta.clone().detach().numpy(),
            U_KL_trace = U_KL_trace + (u_KL_opt,)
        print(loss_u_theta)
        u_KL.requires_grad_(False)
        para_theta.requires_grad_(False)
        lognoisescale.requires_grad_(False)

        p = self.y_obs.shape[1]
        sigma_e_sq_MAP = torch.zeros(p)
        for i in range(p):
            lognoisescale[i]=torch.log(self.GP_Components[i]['noisescale'].double())
            self.GP_Components[0]['noisescale'] = torch.max (torch.exp(lognoisescale_opt[i]), 1e-6 * self.GP_Components[i]['outputscale'])
            sigma_e_sq_MAP[i] = self.GP_Components[i]['noisescale']
        u_opt = torch.empty(self.n_I,self.p).double()
        
        for i in range(self.p):
            u_opt[:,i] = GP_Trans_u_Lu[i]['u_mean_I'] + GP_Trans_u_Lu[i]['u_KL_to_u'] @ u_KL_opt[:,i]
            #u_opt[:,i] = self.GP_Components[i]['mean'] + GP_Trans_u_Lu[i]['u_KL_to_u'] @ u_KL_opt[:,i]
        theta_err=(torch.mean((theta_opt-self.PDE_Model.theta_true).square())).sqrt().clone()
        theta_err_relative=(torch.mean(((theta_opt-self.PDE_Model.theta_true)/self.PDE_Model.theta_true).square())).sqrt().clone()
        print('Estimated parameter:', (theta_opt.clone().detach()).numpy(), 'True parameter:',self.PDE_Model.theta_true.numpy(), 'Error of theta:', theta_err,'relative err',theta_err_relative)
        time1 = time.time() - time0
        map_est={
            'u_KL_trace':U_KL_trace,
            'theta_err': theta_err, 
            'theta_err_relative':theta_err_relative,
            'sigma_e_sq_MAP': sigma_e_sq_MAP, 
            'theta_MAP' : theta_opt.clone().detach(), 
            'u_MAP' : u_opt, 
            'u_KL_MAP': u_KL_opt,
            'time': time1
        }
        return (map_est)

    def Loss_Theta_Marginal(self, theta, u_current, Lu_GP_KL, GP_Trans_u_Lu, Lu_GP, pde_parameter=4):
        if pde_parameter <= 3:
            lkh = torch.zeros(self.p)
            Lu_PDE = self.PDE_Model.Source(self.x_I, u_current, theta)
            for i in range(self.p):
                Lu_Error = Lu_PDE[:,i] - Lu_GP[:,i]
                temp =  0.5 * Lu_Error @ GP_Trans_u_Lu[i]['Kinv'] @ Lu_Error.T
                lkh[i] = temp
            theta_loss = torch.sum(lkh).detach()
            return(theta_loss.numpy())
        elif self.pde_operator == 4 or self.pde_operator == 5 :
            Lu_PDE = self.PDE_Model.Source(self.x_I, u_current, theta)
            Lu_Error = Lu_PDE - Lu_GP
            Lu_Error=Lu_Error[:,2]
            lkh =  0.5 * Lu_Error @ self.LKL33_inv @ Lu_Error.T
            theta_loss = lkh.detach()
            return(theta_loss.numpy())
        elif self.pde_operator == 6:
            Lu_PDE = self.PDE_Model.Source(self.x_I, u_current, theta)
            Lu_Error = Lu_PDE - Lu_GP
            #lkh = Lu_Error[:,2] @ self.LKL_U_inv_margin @ Lu_Error[:,2] + Lu_Error[:,3] @ self.LKL_V_inv_margin @ Lu_Error[:,3]
            lkh =  torch.mean(torch.square(Lu_Error[:,2:3]))
            theta_loss = lkh.detach()
            return(theta_loss.numpy())
        elif self.pde_operator==7:
            Lu_PDE = self.PDE_Model.Source(self.x_I, u_current, theta)
            Lu_Error = Lu_PDE - Lu_GP
            lkh =  torch.mean(torch.square(Lu_Error[:,2]))
            theta_loss = lkh.detach()
            return(theta_loss.numpy())            
        elif self.pde_operator==8:
            Lu_PDE = self.PDE_Model.Source(self.x_I, u_current, theta)
            Lu_Error = Lu_PDE - Lu_GP
            lkh =  torch.mean(torch.square(Lu_Error[:,3]))
            theta_loss = lkh.detach()
            return(theta_loss.numpy())         

    def Minus_Log_Posterior(self, u_KL, para_theta, logsigma = None):
        if self.pde_operator <=3:
            u = torch.empty(self.n_I,self.p).double()
            #Lu_GP_KL = torch.empty(self.M_Lu_KL,self.p).double()
            Lu_GP= torch.empty(self.n_I,self.p).double()
            for i in range(self.p):
                u[:,i] = self.GP_Trans_u_Lu[i]['u_mean_I'] + self.GP_Trans_u_Lu[i]['u_KL_to_u'] @ u_KL[:,i]
                #Lu_GP_KL[:,i] =self.GP_Trans_u_Lu[i]['u_KL_to_Lu_KL'] @ u[:,i]
            Lu_PDE = self.PDE_Model.Source(self.x_I, u, para_theta)
            lkh = torch.zeros((self.p, 3))
            for i in range(self.p):
                outputscale = self.GP_Components[i]['outputscale']
                # p(X[I] = x[I]) = P(U[I] = u[I])
                lkh[i,0] = -0.5 * torch.sum(torch.square(u_KL[:,i]))
                # p(Y[I] = y[I] | X[I] = x[I])
                noisescale = self.noisy_known* (self.GP_Components[i]['noisescale'].clone()) + (1-self.noisy_known) * torch.max(torch.exp(2 * logsigma[i]), 1e-6 * outputscale)
                lkh[i,1] = -0.5 / noisescale * torch.sum ( torch.square(u[self.aIdx,i]-self.y_obs[:,i])) - 0.5 * self.n_obs * torch.log(noisescale) # - 1e6 * (noisescale < 1e-6 * self.GP_Components[i]['outputscale']) * (-torch.log(noisescale))
                # p(X'[I]=f(x[I],theta)|X(I)=x(I))
                Lu_GP[:,i] =self.GP_Trans_u_Lu[i]['Lu_mean_I']  + self.GP_Trans_u_Lu[i]['u_KL_to_Lu_GP'] @ u_KL[:,i]
                Lu_Error = Lu_PDE[:,i] - Lu_GP[:,i]
                lkh[i,2] =  -0.5 * Lu_Error @ self.GP_Trans_u_Lu[i]['Kinv'] @ Lu_Error.T/outputscale
            #print(lkh)
            lkh[:,1] = lkh[:,1] * self.n_I/self.n_obs
        elif self.pde_operator == 4 or self.pde_operator == 5 :
            u = torch.empty(self.n_I,self.p).double()
            #Lu_GP_KL = torch.empty(self.M_Lu_KL,self.p).double()
            Lu_GP= torch.empty(self.n_I,self.p).double()
            for i in range(self.p):
                u[:,i] = self.GP_Trans_u_Lu[i]['u_mean_I'] + self.GP_Trans_u_Lu[i]['u_KL_to_u'] @ u_KL[:,i]
                #u[:,i] = self.GP_Components[i]['mean'] + self.GP_Trans_u_Lu[i]['u_KL_to_u'] @ u_KL[:,i]
            Lu_GP[:,0] = self.GP_Trans_u_Lu[0]['Lu_mean_I']  + self.GP_Trans_u_Lu[0]['u_KL_to_Lu_GP'] @ u_KL[:,0]
            Lu_GP[:,1] = self.GP_Trans_u_Lu[1]['Lu_mean_I']  + self.GP_Trans_u_Lu[1]['u_KL_to_Lu_GP'] @ u_KL[:,1]
            Lu_GP[:,2] = self.GP_Trans_u_Lu[2]['Lu_mean_I']  + self.GP_Trans_u_Lu[2]['u_KL_to_Lu_GP'] @ u_KL[:,0]
            Lu_PDE = self.PDE_Model.Source(self.x_I, u, para_theta)
            lkh = torch.zeros((1, 3))
            lkh[0,0] = -0.5 * torch.sum(torch.square(u_KL)) 
            outputscale = self.GP_Components[0]['outputscale']    
            noisescale = self.noisy_known* (self.GP_Components[0]['noisescale'].clone()) + (1-self.noisy_known) * torch.max(torch.exp(2 * logsigma[0]), 1e-6 * outputscale)
            #lkh[0,1] = -0.5 / noisescale * torch.sum ( torch.square(u[self.aIdx,0]-self.y_obs[:,0])) - 0.5 * self.n_obs * torch.log(noisescale) # - 1e6 * (noisescale < 1e-6 * self.GP_Components[i]['outputscale']) * (-torch.log(noisescale))
            lkh[0,1] = -0.5 / noisescale * torch.sum ( torch.square(u[self.aIdx[0:self.n_obs-self.n_bound],0]-self.y_obs[0:self.n_obs-self.n_bound,0])) - 0.5 * (self.n_obs-self.n_bound) * torch.log(noisescale) -0.5 / (1e-6*self.GP_Components[0]['outputscale']) * torch.sum ( torch.square(u[self.aIdx[self.n_obs-self.n_bound:self.n_obs],0]-self.y_obs[self.n_obs-self.n_bound:self.n_obs,0]))# - 1e6 * (noisescale < 1e-6 * self.GP_Components[i]['outputscale']) * (-torch.log(noisescale))
            Lu_Error = Lu_PDE - Lu_GP
            #lkh[0,2] =  -0.5 * (Lu_Error[:,0] @ self.LKL11_inv @ Lu_Error[:,0].T /self.GP_Components[0]['outputscale']+Lu_Error[:,1] @ self.LKL22_inv @ Lu_Error[:,1].T /self.GP_Components[1]['outputscale']+Lu_Error[:,2] @ self.LKL33_inv @ Lu_Error[:,2].T /self.GP_Components[0]['outputscale'])
            lkh[0,2] =  -0.5 * (torch.cat((Lu_Error[:,0],Lu_Error[:,2])) @ self.LKL1313_inv @ torch.cat((Lu_Error[:,0],Lu_Error[:,2])).T /self.GP_Components[0]['outputscale']+Lu_Error[:,1] @ self.LKL22_inv @ Lu_Error[:,1].T /self.GP_Components[1]['outputscale'])
            #Lu_Error = Lu_Error.T.reshape(-1,1).squeeze()
            #lkh[0,2] =  -0.5 * Lu_Error @ self.LKL_all_inv @ Lu_Error.T
            #lkh[0,2] = -torch.mean(torch.square(Lu_Error))
            lkh[:,1] = 3 * lkh[:,1] * self.n_I/self.n_obs
        elif self.pde_operator == 6 :
            u = torch.empty(self.n_I,self.p).double()
            Lu_GP= torch.empty(self.n_I,self.p).double()
            for i in range(self.p):
                u[:,i] = self.GP_Trans_u_Lu[i]['u_mean_I'] + self.GP_Trans_u_Lu[i]['u_KL_to_u'] @ u_KL[:,i]
                Lu_GP[:,i] = self.GP_Trans_u_Lu[i]['Lu_mean_I']  + self.GP_Trans_u_Lu[i]['u_to_Lu_GP'] @ (u[:,i%2]-self.GP_Components[i%2]['mean'])
            Lu_PDE = self.PDE_Model.Source(self.x_I, u, para_theta)
            lkh = torch.zeros((1, 3))
            lkh[0,0] = -0.5 * torch.sum(torch.square(u_KL)) 
            outputscale = self.GP_Components[0]['outputscale']    
            noisescale = self.noisy_known* (self.GP_Components[0]['noisescale'].clone()) + (1-self.noisy_known) * torch.max(torch.exp(2 * logsigma[0]), 1e-6 * outputscale)
            lkh[0,1] = -0.5 / noisescale * torch.sum ( torch.square(u[self.aIdx[0:self.n_obs-self.n_bound],0]-self.y_obs[0:self.n_obs-self.n_bound,0])) - 0.5 * (self.n_obs-self.n_bound) * torch.log(noisescale) -0.5 / (1e-6*self.GP_Components[0]['outputscale']) * torch.sum ( torch.square(u[self.aIdx[self.n_obs-self.n_bound:self.n_obs],0]-self.y_obs[self.n_obs-self.n_bound:self.n_obs,0]))
            outputscale = self.GP_Components[1]['outputscale']
            noisescale = self.noisy_known* (self.GP_Components[1]['noisescale'].clone()) + (1-self.noisy_known) * torch.max(torch.exp(2 * logsigma[1]), 1e-6 * outputscale)
            lkh[0,1] = lkh[0,1] -0.5 / noisescale * torch.sum ( torch.square(u[self.aIdx[0:self.n_obs-self.n_bound],1]-self.y_obs[0:self.n_obs-self.n_bound,1])) - 0.5 * (self.n_obs-self.n_bound) * torch.log(noisescale) -0.5 / (1e-6*self.GP_Components[1]['outputscale']) * torch.sum ( torch.square(u[self.aIdx[self.n_obs-self.n_bound:self.n_obs],1]-self.y_obs[self.n_obs-self.n_bound:self.n_obs,1]))
            Lu_Error = Lu_PDE - Lu_GP
            lkh[0,2] =  -0.5 * torch.cat((Lu_Error[:,0],Lu_Error[:,2])) @ self.LKL_U_inv @ torch.cat((Lu_Error[:,0],Lu_Error[:,2])).T /self.GP_Components[0]['outputscale']
            lkh[0,2] =  lkh[0,2] -0.5 * torch.cat((Lu_Error[:,1],Lu_Error[:,3])) @ self.LKL_V_inv @ torch.cat((Lu_Error[:,1],Lu_Error[:,3])).T /self.GP_Components[1]['outputscale']
            lkh[:,1] = 2 * lkh[:,1] * self.n_I/self.n_obs
        elif self.pde_operator == 7 :
            u = torch.empty(self.n_I,self.p).double()
            Lu_GP= torch.empty(self.n_I,self.p).double()
            for i in range(self.p):
                u[:,i] = self.GP_Trans_u_Lu[i]['u_mean_I'] + self.GP_Trans_u_Lu[i]['u_KL_to_u'] @ u_KL[:,i]
                Lu_GP[:,i] = self.GP_Trans_u_Lu[i]['Lu_mean_I']  + self.GP_Trans_u_Lu[i]['u_to_Lu_GP'] @ (u[:,i%2]-self.GP_Components[i%2]['mean'])
            Lu_PDE = self.PDE_Model.Source(self.x_I, u, para_theta)
            lkh = torch.zeros((1, 3))
            #print(torch.mean((u[:,1]-self.V_test).square()))
            lkh[0,0] = -0.5 * torch.sum(torch.square(u_KL))# - (para_theta[0]<0) * 1e6 * torch.exp(- 100* para_theta[0])
            outputscale = self.GP_Components[0]['outputscale']    
            noisescale = self.noisy_known* (self.GP_Components[0]['noisescale'].clone()) + (1-self.noisy_known) * torch.max(torch.exp(2 * logsigma[0]), 1e-6 * outputscale)
            lkh[0,1] = -0.5 / noisescale * torch.sum ( torch.square(u[self.aIdx[0:self.n_obs-self.n_bound],0]-self.y_obs[0:self.n_obs-self.n_bound,0])) - 0.5 * (self.n_obs-self.n_bound) * torch.log(noisescale) -0.5 / (1e-6*self.GP_Components[0]['outputscale']) * torch.sum ( torch.square(u[self.aIdx[self.n_obs-self.n_bound:self.n_obs],0]-self.y_obs[self.n_obs-self.n_bound:self.n_obs,0]))
            Lu_Error = Lu_PDE - Lu_GP
            lkh[0,2] =  -0.5 * torch.cat((Lu_Error[:,0],Lu_Error[:,2])) @ self.LKL_U_inv @ torch.cat((Lu_Error[:,0],Lu_Error[:,2])).T /self.GP_Components[0]['outputscale']
            lkh[0,2] =  lkh[0,2] -0.5 * torch.cat((Lu_Error[:,1],Lu_Error[:,3])) @ self.LKL_V_inv @ torch.cat((Lu_Error[:,1],Lu_Error[:,3])).T /self.GP_Components[1]['outputscale']
            lkh[:,1] = 4 * lkh[:,1] * self.n_I/self.n_obs
            #print(torch.mean(torch.square(self.V_test-u[:,1])))
        elif self.pde_operator == 8 :
            u = torch.empty(self.n_I,self.p).double()
            Lu_GP= torch.empty(self.n_I,self.p).double()
            for i in range(self.p):
                u[:,i] = self.GP_Trans_u_Lu[i]['u_mean_I'] + self.GP_Trans_u_Lu[i]['u_KL_to_u'] @ u_KL[:,i]
                Lu_GP[:,i] = self.GP_Trans_u_Lu[i]['Lu_mean_I']  + self.GP_Trans_u_Lu[i]['u_to_Lu_GP'] @ (u[:,i%2]-self.GP_Components[i%2]['mean'])
            Lu_PDE = self.PDE_Model.Source(self.x_I, u, para_theta)
            lkh = torch.zeros((1, 3))
            lkh[0,0] = -0.5 * torch.sum(torch.square(u_KL)) 
            outputscale = self.GP_Components[0]['outputscale']    
            noisescale = self.noisy_known* (self.GP_Components[1]['noisescale'].clone()) + (1-self.noisy_known) * torch.max(torch.exp(2 * logsigma[1]), 1e-6 * outputscale)
            lkh[0,1] = -0.5 / noisescale * torch.sum ( torch.square(u[self.aIdx[1:self.n_obs-self.n_bound],0]-self.y_obs[1:self.n_obs-self.n_bound,0])) - 0.5 * (self.n_obs-self.n_bound) * torch.log(noisescale) -0.5 / (1e-6*self.GP_Components[0]['outputscale']) * torch.sum ( torch.square(u[self.aIdx[self.n_obs-self.n_bound:self.n_obs],0]-self.y_obs[self.n_obs-self.n_bound:self.n_obs,0]))
            Lu_Error = Lu_PDE - Lu_GP
            lkh[0,2] =  -0.5 * torch.cat((Lu_Error[:,0],Lu_Error[:,2])) @ self.LKL_U_inv @ torch.cat((Lu_Error[:,0],Lu_Error[:,2])).T /self.GP_Components[0]['outputscale']
            lkh[0,2] =  lkh[0,2] -0.5 * torch.cat((Lu_Error[:,1],Lu_Error[:,3])) @ self.LKL_V_inv @ torch.cat((Lu_Error[:,1],Lu_Error[:,3])).T /self.GP_Components[1]['outputscale']
            lkh[:,1] = 4 * lkh[:,1] * self.n_I/self.n_obs
            #print(torch.mean(torch.square(self.U_test-u[:,1])))
        return (-torch.sum(lkh))
    
    def Sample_Using_HMC(self, n_epoch = 5000, lsteps=100, epsilon=1e-5, n_samples=20000, Map_Estimation = None, Normal_Approxi_Only = False):
        if Map_Estimation == None : Map_Estimation=self.map(nEpoch = n_epoch)
        self.Map_Estimation = Map_Estimation
        log_sigma=torch.log(Map_Estimation['sigma_e_sq_MAP']).double() / 2
        u_KL=Map_Estimation['u_KL_MAP']
        theta=Map_Estimation['theta_MAP']
        self.sampler = HMC.Posterior_Density_Inference(self.Minus_Log_Posterior, u_KL, theta, log_sigma, u_KL.shape, theta.shape, log_sigma.shape, noisy_known=self.noisy_known, lsteps=lsteps, epsilon=epsilon, n_samples=n_samples)
        self.Normal_Approxi_Only = Normal_Approxi_Only
        self.Posterior_PDE_NA = self.sampler.Normal_Approximation()
        if Normal_Approxi_Only is True:
            return (self.Posterior_PDE_NA, self.Map_Estimation)
        self.HMC_sample = self.sampler.Sampling()
        return (self.HMC_sample, self.Posterior_PDE_NA, self.Map_Estimation)
    
    def Posterior_Summary(self, alpha =0.05, draw_posterior_sample = False):
        Var = torch.diag(self.Posterior_PDE_NA['variance'])
        Mean = self.Posterior_PDE_NA['mean']
        mean_theta = Mean[-(self.d_theta+1):-1]
        var_theta = Var[-(self.d_theta+1):-1]
        theta_CI_NA = torch.zeros(2, mean_theta.shape[0])
        theta_CI_NA[0,:] = mean_theta - scipy.stats.norm.cdf(alpha)*torch.sqrt(var_theta)
        theta_CI_NA[1,:] = mean_theta + scipy.stats.norm.cdf(alpha)*torch.sqrt(var_theta)
        if self.Normal_Approxi_Only is True :
            return (theta_CI_NA)
        chain=self.HMC_sample['samples']
        sampler=self.sampler
        x_pred=self.PDE_Model.x_pred
        GP_Trans_u_Lu = self.GP_Trans_u_Lu
        N_chain = chain.shape[0]
        Err_u_chain = torch.zeros(N_chain)
        u_pred = torch.zeros((N_chain, x_pred.shape[0],self.p))
        theta_post = torch.zeros((N_chain, sampler.theta_shape[0]))
        sigmasq_post = torch.zeros((N_chain, 1, self.p))
        for s in range (N_chain):
            sample = torch.tensor(chain[s,:])
            u_KL_current, theta_current, logsigma_current = sampler.devectorize(sample, sampler.u_KL_shape, sampler.theta_shape, sampler.sigma_shape)
            u_KL_current = u_KL_current.clone()
            if draw_posterior_sample is True:
                u_current = torch.empty(self.n_I,self.p).double()
                for i in range(self.p):
                    u_current[:,i] = GP_Trans_u_Lu[i]['u_mean_I'] + GP_Trans_u_Lu[i]['u_KL_to_u'] @ u_KL_current[:,i]
                Err_u_chain[s], u_pred_temp = self.Predict_err(u_current)[0:2]
                u_pred[s] = u_pred_temp.clone()
            theta_post[s,:] = theta_current.clone()
            sigmasq_post[s,:] = torch.exp(2 * logsigma_current).clone()
        self.posterior_summary = {
            'u_pred': u_pred,
            'u_pred_true': self.PDE_Model.y_pred,
            'theta_posterior':theta_post,
            'sigmasq_posterior':sigmasq_post,
            'u_err_posterior':Err_u_chain
        }
        burn_in = np.int(N_chain*0.1)
        theta_post = theta_post[burn_in:N_chain,:]
        theta_CI = self.Credible_Interval(theta_post, alpha = 0.05)
        return (theta_CI, theta_CI_NA)

    def Credible_Interval(self, samples, alpha = 0.05):
        N = samples.shape[0]
        d = samples.shape[1]
        L = np.int(N*alpha/2)
        U = N - np.int(N*alpha/2)
        CI = torch.zeros(2, d)
        CI[0,:]=samples.sort(0).values[L,:]
        CI[1,:]=samples.sort(0).values[U,:]
        return (CI)

    def Predition_Solution(self , U_KL_trace):
        if self.pde_operator ==5 :
            x = torch.linspace(0,1,21)
            x_pred= torch.cat((0.1 * torch.ones(21,1),x.reshape(-1,1)),1)
        elif self.pde_operator <= 2:
            x = torch.linspace(0,1,21)
            x_pred= torch.cat((torch.ones(21,1),x.reshape(-1,1)),1)
        elif self.pde_operator == 3:
            x, y = torch.meshgrid(torch.linspace(0,1,11),torch.linspace(0,1,11))
            x = torch.linspace(0,1,21)
            y = torch.linspace(0,1,21)
            x = x.reshape(-1,1)
            y = y.reshape(-1,1)
            x_pred= torch.cat((0.15 * torch.ones(x.shape[0],1),x,y),1)
        elif self.pde_operator == 4:
            x = torch.linspace(0,1,21)
            x_pred= torch.cat((20 * torch.ones(21,1),40 * x.reshape(-1,1)),1)
        elif self.pde_operator >= 6:
            x, y = torch.meshgrid(torch.linspace(0,1,11),torch.linspace(0,1,11))
            x = torch.linspace(0,1,21)
            y = torch.linspace(0,1,21)
            x = x.reshape(-1,1)
            y = y.reshape(-1,1)
            x_pred= torch.cat((0.15 * torch.ones(x.shape[0],1),x,y),1)
        u_pred_trace = ()
        num = 0
        u_true = self.PDE_Model.True_Solution(x_pred)
        p = 1 * (self.pde_operator <= 5 ) + 2 * (self.pde_operator > 6 ) 
        Cxy = ()
        for i in range (p):
            kernel = self.GP_Components[i]['kernel']
            Cxy= Cxy + (kernel.K(x_pred, self.x_I),)
        for u_KL in (U_KL_trace):
            num=num+1
            #u_KL = U_KL_trace[i]
            u = torch.empty(self.n_I,self.p).double()
            for i in range(self.p):
                u[:,i] = self.GP_Trans_u_Lu[i]['u_mean_I'] + self.GP_Trans_u_Lu[i]['u_KL_to_u'] @ u_KL[:,i]           
            u_pred = torch.zeros(x_pred.shape[0],p)
            for i in range (p):
                mean = self.GP_Components[i]['mean']
                C_inv = self.GP_Models[i].K_inv
                u_pred[:,i] = mean+ Cxy[i] @ C_inv @ (u[:,i] - mean)
            err_u = (torch.mean(torch.square(u_pred-u_true))).sqrt()
            u_pred_trace = u_pred_trace + (u_pred,)
        return (x_pred, u_pred_trace, u_true)

    def _Pre_Process(self):
        # obtain features from GP_Components
        if self.pde_operator <=3 :
            # obtain features from GP_Components
            GP_Trans_u_Lu = []
            u_KL = torch.empty(self.n_I, self.p).double()
            u = torch.empty(self.n_I, self.p).double()
            Lu_GP = torch.empty(self.n_I, self.p).double()
            Lu_GP_KL = torch.empty(self.n_I, self.p).double()
            M_all=0
            M_Lu_all=0
            for i in range(self.p):
                # get observation data
                y_all_obs = torch.cat((self.y_obs[:,i],self.y_bound[:,i]),0)
                x_all_obs = torch.cat((self.x_obs,self.x_bound),0)
                theta=self.PDE_Model.para_theta
                # get hyperparameters
                kernel = self.GP_Components[i]['kernel']
                mean = self.GP_Components[i]['mean']
                outputscale = self.GP_Components[i]['outputscale']
                # Compute GP prior covariance matrix
                Cbb = kernel.K(self.x_bound) + 1e-6 * torch.eye(self.n_bound)
                self.invCbb=torch.inverse(Cbb)
                self.GP_Models[i].invCbb=self.invCbb
                CbI = kernel.K(self.x_bound,self.x_I)
                CIb = kernel.K(self.x_I,self.x_bound)
                self.Cxx = kernel.K(self.x_I) + 1e-6 * torch.eye(self.n_I) - CIb @ self.invCbb @ CbI
                u_mean_I=mean + CIb @ self.invCbb @ (self.y_bound[:,i].T-mean)
                P, V, Q=torch.svd(self.Cxx)
                if self. KL is False :
                    M_u_KL=self.n_I
                else:
                    #M=sum(V >2 *1e-6)
                    M_u_KL=sum(np.cumsum(V) < (1-1e-6)* torch.sum(V))
                    print('number of KL basis:', M_u_KL)
                Trans_u_to_u_KL = torch.diag(torch.pow(V , -1/2)) @ P.T
                Trans_u_KL_to_u = P @ torch.diag(torch.pow(V , 1/2))
                Trans_u_to_u_KL = Trans_u_to_u_KL[0:M_u_KL,:] #u_to_u_KL
                Trans_u_KL_to_u = Trans_u_KL_to_u[:,0:M_u_KL] #u_KL_to_u
                '''
                L_chol = torch.linalg.cholesky(self.Cxx)
                Trans_u_to_u_KL = torch.inverse(L_chol)
                Trans_u_KL_to_u = L_chol
                '''
                self.Cinv = torch.inverse(self.Cxx)
                self.GP_Models[i].Cinv=self.Cinv
                # obtain initial values
                C_Ia = kernel.K(self.x_I, x_all_obs)
                S = C_Ia @ torch.inverse(self.GP_Components[i]['corr_data'])
                u[:,i] = mean + S @ (y_all_obs - mean)
                # obtain PDE information
                #GP_temp=self.GP_Models[i]
                LKL_II = kernel.LKL(self.x_I)
                LK_II = kernel.LK(self.x_I)
                KL_II = kernel.KL(self.x_I)
                #LKL_II, LK_II, KL_II=GP_temp.Calculating_Cov_Theta(GP_temp.GP_LKL_Component_II, GP_temp.GP_LK_Component_II, GP_temp.GP_KL_Component_II, theta)
                LK_Ib = kernel.LK(self.x_I, self.x_bound) #
                #LK_Ib = GP_temp.Calculating_Cov_Theta_LK(GP_temp.GP_LK_Component_Ib, theta)
                KL_bI =  kernel.KL(self.x_bound, self.x_I) #
                #KL_bI = GP_temp.Calculating_Cov_Theta_KL(GP_temp.GP_KL_Component_bI, theta)
                K_bI = kernel.K(self.x_bound, self.x_I) #
                #K_bI = GP_temp.Calculating_Cov_Theta_K(GP_temp.GP_Cov_Component_bI)
                K_Ib = kernel.K(self.x_I, self.x_bound) #
                #K_Ib = GP_temp.Calculating_Cov_Theta_K(GP_temp.GP_Cov_Component_Ib)
                dCdx1 = LK_II- LK_Ib @ self.invCbb @ K_bI
                dCdx2 = KL_II- K_Ib @ self.invCbb @ KL_bI
                d2Cdx1dx2 = LKL_II+ 1e-6 * torch.eye(self.n_I)- LK_Ib @ self.invCbb @ KL_bI
                K = d2Cdx1dx2 - dCdx1 @ self.Cinv @ dCdx2 
                Lu_mean_I = LK_Ib @ self.invCbb @ (self.y_bound[:,i].T-mean)
                P_L, V_L, Q = torch.svd(K)
                M_Lu_KL=self.n_I
                Trans_Lu_to_Lu_KL = torch.diag(torch.pow(V_L , -1/2))@P_L.T
                Trans_Lu_KL_to_Lu = P_L @ torch.diag(torch.pow(V_L , 1/2))
                Trans_Lu_to_Lu_KL = Trans_Lu_to_Lu_KL[0:M_Lu_KL,:]# trans Lu_GP to Lu_GP_KL
                Trans_Lu_KL_to_Lu = Trans_Lu_KL_to_Lu[:,0:M_Lu_KL]          
                Kinv = torch.inverse(K)
                
                u_KL[0:M_u_KL,i] = Trans_u_to_u_KL @ (u[:,i] - u_mean_I) / np.sqrt(outputscale)
                Trans_u_KL_to_Lu=dCdx1 @ self.Cinv @ Trans_u_KL_to_u
                Lu_GP[:,i] = Lu_mean_I + dCdx1 @ self.Cinv @ (u[:,i] - u_mean_I) 
                Lu_GP_KL [0:M_Lu_KL,i]= Trans_Lu_to_Lu_KL @ Lu_GP[:,i]  / np.sqrt(outputscale)
                Trans_u_KL_to_Lu_KL = Trans_Lu_to_Lu_KL @ Trans_u_KL_to_Lu # trans u to Lu_GP_KL
                GP_Trans_u_Lu.append({
                    'u_mean_I' : u_mean_I,
                    'Lu_mean_I' : Lu_mean_I,
                    'u_KL_to_u' : Trans_u_KL_to_u * np.sqrt(outputscale),
                    'u_to_u_KL' : Trans_u_to_u_KL / np.sqrt(outputscale), 
                    'u_KL_to_Lu_GP' : np.sqrt(outputscale) * Trans_u_KL_to_Lu, 
                    'Kinv' : Kinv, 
                    'K_IbinvKbb' : kernel.K(self.x_I, self.x_bound) @ self.invCbb, 
                    'u_KL_to_Lu_KL' : Trans_u_KL_to_Lu_KL, 
                    'Lu_to_Lu_KL' : Trans_Lu_to_Lu_KL / np.sqrt(outputscale),
                    'M_Lu_KL' : M_Lu_KL,
                    'mean_coefficient' : self.invCbb @ (self.y_bound[:,i].T-mean)
                    })
                M_all=np.max((M_all, M_u_KL))
            u_KL=u_KL[0:M_all,:].clone().detach()
            Lu_GP_KL=Lu_GP_KL[0:M_Lu_KL,:]
            self.GP_Trans_u_Lu=GP_Trans_u_Lu
            self.M_Lu_KL=M_Lu_KL
            return (u_KL, u, Lu_GP, Lu_GP_KL, GP_Trans_u_Lu)
        elif self.pde_operator == 4 or self.pde_operator == 5: 
            nu=self.PDE_Model.nu
            GP_model = self.GP_Models[0]
            y_obs_all = self.y_obs#torch.cat((self.y_obs,self.y_bound),0)
            n_obs_all = y_obs_all.shape[0]
            x_obs_all = self.x_obs#torch.cat((self.x_obs,self.x_bound),0)
            kernel = GP_model.kernel
            nugget_gp = torch.cat(((GP_model.noisescale/GP_model.outputscale)*torch.ones(self.n_obs-self.n_bound),1e-6 * torch.ones(self.n_bound))) #torch.cat((GP_model.noisescale*torch.ones(self.n_obs),1e-6 * torch.ones(n_obs_all-self.n_obs)))
            C = kernel.K(x_obs_all) + torch.diag(nugget_gp)
            U_to_U = kernel.K(self.x_I,x_obs_all)
            U = GP_model.mean + U_to_U @ torch.linalg.inv(C) @ (y_obs_all-GP_model.mean)
            C = kernel.K(self.x_I) + 1e-6 * torch.eye(self.n_I)
            U_to_U_x = kernel.LK(self.x_I)[0]
            U_x = U_to_U_x @ torch.linalg.inv(C) @ (U-GP_model.mean)
            U_to_U_xx = kernel.LK(self.x_I)[1]
            U_xx = U_to_U_xx @ torch.linalg.inv(C) @ (U-GP_model.mean)
            U_to_U_t = kernel.LK(self.x_I)[2]
            U_t = U_to_U_t @ torch.linalg.inv(C) @ (U-GP_model.mean)
            GP_model1=GP_processing.GP_modeling(self.PDE_Model, noisy = False, nu=nu, noisy_known=True)
            GP_model1.Train_GP(self.PDE_Model,self.x_I, U_x, noisy = False)
            self.GP_Components.append({
                #'aIdx':aIdx, # non-missing data index
                'mean':GP_model1.mean,
                'kernel':GP_model1.kernel,
                'outputscale':GP_model1.outputscale,
                'noisescale':GP_model1.noisescale
            })
            self.GP_Models.append(GP_model1)
            GP_model2=GP_processing.GP_modeling(self.PDE_Model, noisy = False, nu=nu, noisy_known=True)
            GP_model2.Train_GP(self.PDE_Model,self.x_I, U_xx, noisy = False)
            self.GP_Components.append({
                #'aIdx':aIdx, # non-missing data index
                'mean':GP_model2.mean,
                'kernel':GP_model2.kernel,
                'outputscale':GP_model2.outputscale,
                'noisescale':GP_model2.noisescale
            })
            self.GP_Models.append(GP_model2)
            U = torch.cat((U,U_x,U_xx),1)
            kernel0 = self.GP_Components[0]['kernel']
            kernel1 = self.GP_Components[1]['kernel'] #K_II, LK1, LK2, LK3, KL1, KL2, KL3, LKL11, LKL12, LKL13, LKL21, LKL22, LKL23, LKL31, LKL32, LKL33 = 
            LK1, LK2, LK3 = kernel0.LK(self.x_I)
            KL1, KL2, KL3 = kernel0.KL(self.x_I)
            LKL11, LKL12, LKL13, LKL21, LKL22, LKL23, LKL31, LKL32, LKL33 = kernel0.LKL(self.x_I)
            K_II = kernel0.K(self.x_I)
            self.LKL33 = LKL33 + 1e-6 * torch.eye(LKL33.shape[0]) - torch.cat((LK3,LKL31),1) @ torch.linalg.inv( torch.cat((torch.cat((K_II,KL1),1),torch.cat((LK1,LKL11),1)),0) + 1e-6 * torch.eye(2 * self.n_I)) @ torch.cat((KL3,LKL13),0)
            self.LKL33_inv = torch.linalg.inv(self.LKL33)
            self.LKL11 = LKL11 + 1e-6 * torch.eye(LKL11.shape[0]) - LK1@ torch.linalg.inv(K_II + 1e-6 * torch.eye(K_II.shape[0])) @ KL1
            self.LKL11_inv = torch.linalg.inv(self.LKL11)
            self.LKL22 = kernel1.LKL(self.x_I)[0] + 1e-6 * torch.eye(self.n_I) - kernel1.LK(self.x_I)[0] @ torch.linalg.inv(kernel1.K(self.x_I) +1e-6 * torch.eye(self.n_I)) @ kernel1.KL(self.x_I)[0]
            self.LKL22_inv = torch.linalg.inv(self.LKL22)
            self.LKL1313 = torch.cat((torch.cat((LKL11,LKL13),1),torch.cat((LKL31,LKL33),1)),0) + 1e-6 * torch.eye(2*self.n_I)- torch.cat((LK1,LK3),0) @ torch.linalg.inv(K_II + 1e-6 * torch.eye(self.n_I)) @ torch.cat((KL1,KL3),1)
            self.LKL1313_inv = torch.linalg.inv(self.LKL1313)
            self.p = 3
            GP_Trans_u_Lu = []
            u_KL = torch.empty(self.n_I, self.p).double()
            Lu_GP = torch.empty(self.n_I, self.p).double()
            Lu_GP_KL = torch.empty(self.n_I, self.p).double()
            M_KL_all = torch.zeros(self.p)
            M_KL_Lu_all = 0
            x_all = torch.cat((self.x_I,self.x_bound))
            for i in range(self.p):
                kernel = self.GP_Components[i]['kernel']
                outputscale = self.GP_Components[i]['outputscale']
                # Compute GP prior covariance matrix
                if i==0: 
                    Cbb = kernel.K(self.x_bound) + 1e-6 * torch.eye(self.x_bound.shape[0])
                    #print('min eigens', torch.min(torch.real(torch.linalg.eig(Cbb).eigenvalues)))
                    invCbb = torch.linalg.inv(Cbb)
                    CbI = kernel.K(self.x_bound, self.x_I)
                    CIb = kernel.K(self.x_I, self.x_bound)
                    K_II = kernel.K(self.x_I, self.x_I) + 1e-6 * torch.eye(self.n_I) #- CIb @ invCbb @ CbI
                    self.test_KII1 = kernel.K(self.x_I, self.x_I) + 1e-6 * torch.eye(self.n_I) - CIb @ invCbb @ CbI
                    #print(self.test_KII1)
                    self.test_KII1_inv = torch.linalg.inv(self.test_KII1)
                    u_mean_I = self.GP_Components[0]['mean'] #+ CIb @ invCbb @ (self.y_bound[:,0].T - self.GP_Components[0]['mean'])
                    u_mean_I_with_b = u_mean_I
                else: 
                    K_II = kernel.K(self.x_I, self.x_I) + 1e-6 * torch.eye(self.n_I)
                    #print(K_II)
                    u_mean_I = self.GP_Components[i]['mean']
                #print('min eigen', torch.min(torch.real(torch.linalg.eig(K_II).eigenvalues)))
                # dimension reduction via KL expansion
                P, V, Q = torch.svd(K_II)
                if self.KL is False :
                    M_u_KL = self.n_I
                else:
                    #M=sum(V >2 *1e-6)
                    M_u_KL = sum(np.cumsum(V) < (1-1e-6)* torch.sum(V))
                    print('number of KL basis:', M_u_KL)
                Trans_u_to_u_KL = torch.diag(torch.pow(V, -1/2)) @ P.T
                Trans_u_KL_to_u = P @ torch.diag(torch.pow(V , 1/2))
                Trans_u_to_u_KL = Trans_u_to_u_KL[0:M_u_KL,:] / np.sqrt(outputscale)#u_to_u_KL
                Trans_u_KL_to_u = Trans_u_KL_to_u[:,0:M_u_KL] * np.sqrt(outputscale)#u_KL_to_u
                '''
                L_chol = torch.linalg.cholesky(K_II)
                Trans_u_to_u_KL = torch.linalg.inv(L_chol)
                Trans_u_KL_to_u = L_chol
                '''
                if i == 1: K_inv = torch.linalg.inv(K_II)
                else:
                    K_II = kernel.K(x_all, x_all) + 1e-6 * torch.eye(x_all.shape[0])
                    K_inv = torch.linalg.inv(K_II)
                self.GP_Models[i].K_inv = K_inv
                # obtain initial values and PDE information
                if i == 1 : 
                    Lu_mean_I = 0
                    LK_II = kernel1.LK(self.x_I)[0]
                else :
                    LK_Ib = kernel0.LK(self.x_I, self.x_bound)[i]
                    Lu_mean_I = 0 #+ LK_Ib @ invCbb @ (self.y_bound[:,0].T-self.GP_Components[0]['mean'])
                    LK_II = kernel0.LK(self.x_I,x_all)[i] #- LK_Ib @ invCbb @ kernel0.K(self.x_bound, self.x_I)
                u_KL[0:M_u_KL,i] = Trans_u_to_u_KL @ (U[:,i] - u_mean_I) 
                if i == 1 : 
                    Trans_u_to_Lu = LK_II @ self.GP_Models[1].K_inv
                    Lu_GP[:,i] = Lu_mean_I + Trans_u_to_Lu @ (U[:,1] - self.GP_Components[1]['mean'])
                    Trans_u_KL_to_Lu = Trans_u_to_Lu @ Trans_u_KL_to_u
                elif i == 0: 
                    Trans_u_to_Lu = LK_II @ self.GP_Models[0].K_inv
                    Lu_GP[:,i] = Lu_mean_I + Trans_u_to_Lu @ (U[:,0] - self.GP_Components[0]['mean'])
                    Trans_u_KL_to_Lu = Trans_u_to_Lu @ Trans_u_KL_to_u
                elif i == 2: 
                    Trans_u_to_Lu = LK_II @ self.GP_Models[0].K_inv
                    Lu_GP[:,i] = Lu_mean_I + Trans_u_to_Lu @ (U[:,0] - self.GP_Components[0]['mean'])
                    Trans_u_KL_to_Lu = Trans_u_to_Lu @ GP_Trans_u_Lu[0]['u_KL_to_u']
                GP_Trans_u_Lu.append({
                    'u_mean_I' : u_mean_I,
                    'Lu_mean_I' : Lu_mean_I,
                    'u_KL_to_u' : Trans_u_KL_to_u,
                    'u_to_u_KL' : Trans_u_to_u_KL , 
                    'u_KL_to_Lu_GP' : Trans_u_KL_to_Lu, 
                    'u_to_Lu_GP' : Trans_u_to_Lu, 
                    })
                M_KL_all[i] = M_u_KL
            #u_KL = u_KL[0:M_KL_all,:].clone().detach()
            self.GP_Trans_u_Lu = GP_Trans_u_Lu
            return (u_KL, U, Lu_GP, Lu_GP_KL, GP_Trans_u_Lu)
        elif self.pde_operator == 6 or self.pde_operator == 7 or self.pde_operator == 8 : 
            nu=self.PDE_Model.nu
            y_obs_all = self.y_obs#torch.cat((self.y_obs,self.y_bound),0)
            n_obs_all = y_obs_all.shape[0]
            x_obs_all = self.x_obs#torch.cat((self.x_obs,self.x_bound),0)
            GP_model = self.GP_Models[0]
            nugget_gp = torch.cat((GP_model.noisescale/GP_model.outputscale *torch.ones(self.n_obs-self.n_bound),1e-6 * torch.ones(self.n_bound))) 
            kernel = GP_model.kernel
            C = kernel.K(x_obs_all) + torch.diag(nugget_gp)
            U = GP_model.mean + kernel.K(self.x_I,x_obs_all) @ torch.linalg.inv(C) @ (y_obs_all[:,0]-GP_model.mean)
            C = kernel.K(self.x_I) + 1e-6 * torch.eye(self.n_I)
            Lap_U = kernel.LK(self.x_I)[0] @ torch.linalg.inv(C) @ (U-GP_model.mean)
            U_t = kernel.LK(self.x_I)[1] @ torch.linalg.inv(C) @ (U-GP_model.mean)

            GP_model = self.GP_Models[1]
            nugget_gp = torch.cat((GP_model.noisescale/GP_model.outputscale * torch.ones(self.n_obs-self.n_bound),1e-6 * torch.ones(self.n_bound))) 
            kernel = GP_model.kernel
            nugget_gp = torch.cat((GP_model.noisescale/GP_model.outputscale * torch.ones(self.n_obs-self.n_bound),1e-6 * torch.ones(self.n_bound)))
            C = kernel.K(x_obs_all) + torch.diag(nugget_gp)
            V = GP_model.mean + kernel.K(self.x_I,x_obs_all) @ torch.linalg.inv(C) @ (y_obs_all[:,1]-GP_model.mean)
            C = kernel.K(self.x_I) + 1e-6 * torch.eye(self.n_I) 
            Lap_V = kernel.LK(self.x_I)[0] @ torch.linalg.inv(C) @ (V-GP_model.mean)
            V_t = kernel.LK(self.x_I)[1] @ torch.linalg.inv(C) @ (V-GP_model.mean)
            U = U.reshape(-1,1)
            Lap_U = Lap_U.reshape(-1,1)
            U_t = U_t.reshape(-1,1)
            V = V.reshape(-1,1)
            Lap_V = Lap_V.reshape(-1,1)
            V_t = V_t.reshape(-1,1)
            GP_model1=GP_processing.GP_modeling(self.PDE_Model, noisy = False, nu=nu, noisy_known=True)
            GP_model1.Train_GP(self.PDE_Model,self.x_I, Lap_U, noisy = False)
            self.GP_Components.append({
                #'aIdx':aIdx, # non-missing data index
                'mean':GP_model1.mean,
                'kernel':GP_model1.kernel,
                'outputscale':GP_model1.outputscale,
                'noisescale':GP_model1.noisescale
            })
            self.GP_Models.append(GP_model1)
            GP_model2=GP_processing.GP_modeling(self.PDE_Model, noisy = False, nu=nu, noisy_known=True)
            GP_model2.Train_GP(self.PDE_Model,self.x_I, Lap_V, noisy = False)
            self.GP_Components.append({
                #'aIdx':aIdx, # non-missing data index
                'mean':GP_model2.mean,
                'kernel':GP_model2.kernel,
                'outputscale':GP_model2.outputscale,
                'noisescale':GP_model2.noisescale
            })
            self.GP_Models.append(GP_model2)
            U = torch.cat((U,V,Lap_U,Lap_V,U_t,V_t),1)
            self.p = 4
            GP_Trans_u_Lu = []
            u_KL = torch.empty(self.n_I, self.p).double()
            Lu_GP = torch.empty(self.n_I, self.p).double()
            Lu_GP_KL = torch.empty(self.n_I, self.p).double()
            M_KL_all = torch.zeros(self.p)
            M_KL_Lu_all = 0
            for i in range(self.p):
                base_u = i % 2
                base_ope = int(i/2)
                kernel = self.GP_Components[i]['kernel']
                outputscale = self.GP_Components[i]['outputscale']
                # Compute GP prior covariance matrix
                K_II = kernel.K(self.x_I, self.x_I) + 1e-6 * torch.eye(self.n_I)
                u_mean_I = self.GP_Components[i]['mean'] 
                # dimension reduction via KL expansion
                P, V, Q = torch.svd(K_II)
                if self.KL is False :
                    M_u_KL = self.n_I
                else:
                    #M=sum(V >2 *1e-6)
                    M_u_KL = sum(np.cumsum(V) < (1-1e-6)* torch.sum(V))
                    print('number of KL basis:', M_u_KL)
                Trans_u_to_u_KL = torch.diag(torch.pow(V, -1/2)) @ P.T
                Trans_u_KL_to_u = P @ torch.diag(torch.pow(V , 1/2))
                Trans_u_to_u_KL = Trans_u_to_u_KL[0:M_u_KL,:] / np.sqrt(outputscale)#u_to_u_KL
                Trans_u_KL_to_u = Trans_u_KL_to_u[:,0:M_u_KL] * np.sqrt(outputscale)#u_KL_to_u
                if i ==0 : Trans_u_KL_to_u_0 = Trans_u_KL_to_u
                '''
                L_chol = torch.linalg.cholesky(K_II)
                Trans_u_to_u_KL = torch.linalg.inv(L_chol)
                Trans_u_KL_to_u = L_chol
                '''
                K_inv = torch.linalg.inv(K_II)
                self.GP_Models[i].K_inv = K_inv
                # obtain initial values and PDE information
                u_KL[0:M_u_KL,i] = Trans_u_to_u_KL @ (U[:,i] - u_mean_I)
                Lu_mean_I = 0
                LK_II = self.GP_Components[base_u]['kernel'].LK(self.x_I)[base_ope]
                Trans_u_to_Lu = LK_II @ self.GP_Models[base_u].K_inv
                Trans_u_KL_to_Lu = Trans_u_to_Lu @ Trans_u_KL_to_u_0
                Lu_GP[:,i] = Lu_mean_I + Trans_u_to_Lu @ (U[:,base_u] - self.GP_Components[base_u]['mean']) 
                GP_Trans_u_Lu.append({
                    'u_mean_I' : u_mean_I,
                    'Lu_mean_I' : Lu_mean_I,
                    'u_KL_to_u' : Trans_u_KL_to_u ,
                    'u_to_u_KL' : Trans_u_to_u_KL , 
                    'u_KL_to_Lu_GP' : Trans_u_KL_to_Lu, 
                    'u_to_Lu_GP' : Trans_u_to_Lu, 
                    })
                M_KL_all[i] = M_u_KL
            #u_KL = u_KL[0:M_KL_all,:].clone().detach()
            self.V_test = U[:,1].clone()
            self.U_test = U[:,0].clone()
            if self.pde_operator ==7 :
                #if self.cheat ==1: 
                # U[:,1] = 1 + 0.8 * self.x_I[:,1]
                #torch.ones(self.n_I)
                GP_Trans_u_Lu[1] = GP_Trans_u_Lu[0] 
                GP_Trans_u_Lu[3] = GP_Trans_u_Lu[2] 
                self.GP_Components[1] = self.GP_Components[0]
                self.GP_Components[3] = self.GP_Components[2]
                self.GP_Models[1] = self.GP_Models[0] 
                self.GP_Models[3] = self.GP_Models[2] 
                u_KL[0:M_u_KL,1] = GP_Trans_u_Lu[1]['u_to_u_KL'] @ (U[:,1] - GP_Trans_u_Lu[1]['u_mean_I'])
                #u_KL[0:M_u_KL,1] = torch.randn(self.n_I)
                #U[:,1] =  GP_Trans_u_Lu[1]['u_mean_I'] + GP_Trans_u_Lu[1]['u_KL_to_u'] @ u_KL[0:M_u_KL,1]
                U[:,3] = GP_Trans_u_Lu[3]['u_mean_I'] + GP_Trans_u_Lu[1]['u_to_Lu_GP'] @ (U[:,1] - GP_Trans_u_Lu[1]['u_mean_I'])
                u_KL[0:M_u_KL,3] = GP_Trans_u_Lu[3]['u_to_u_KL'] @ (U[:,3] - GP_Trans_u_Lu[3]['u_mean_I'])
            elif self.pde_operator ==8 :
                U[:,0] = 2 + 0.25 * self.x_I[:,2]#torch.ones(self.n_I)
                GP_Trans_u_Lu[0] = GP_Trans_u_Lu[1] 
                GP_Trans_u_Lu[2] = GP_Trans_u_Lu[3] 
                self.GP_Components[0] = self.GP_Components[1]
                self.GP_Components[2] = self.GP_Components[3]
                self.GP_Models[0] = self.GP_Models[1] 
                self.GP_Models[2] = self.GP_Models[3] 
                u_KL[0:M_u_KL,0] = GP_Trans_u_Lu[0]['u_to_u_KL'] @ (U[:,0] - GP_Trans_u_Lu[0]['u_mean_I'])
                U[:,2] = GP_Trans_u_Lu[2]['u_mean_I'] + GP_Trans_u_Lu[0]['u_to_Lu_GP'] @ (U[:,0] - GP_Trans_u_Lu[0]['u_mean_I'])
                u_KL[0:M_u_KL,2] = GP_Trans_u_Lu[2]['u_to_u_KL'] @ (U[:,2] - GP_Trans_u_Lu[2]['u_mean_I'])
            self.GP_Trans_u_Lu = GP_Trans_u_Lu
            kernel1 = self.GP_Components[0]['kernel']
            LKL_all = kernel1.LKL(self.x_I)[4]
            LK_all = kernel1.LK(self.x_I)[2]
            KL_all = kernel1.KL(self.x_I)[2]
            self.LKL_U = LKL_all - LK_all @ self.GP_Models[0].K_inv @ KL_all + 1e-6 * torch.eye(2*self.n_I)
            self.LKL_U_inv = torch.linalg.inv(self.LKL_U)
            self.LKL_U_margin = kernel1.LKL(self.x_I)[1] - kernel1.LK(self.x_I)[1] @ self.GP_Models[0].K_inv @ kernel1.KL(self.x_I)[1] + 1e-6 * torch.eye(self.n_I)
            self.LKL_U_inv_margin = torch.linalg.inv(self.LKL_U_margin)
            kernel1 = self.GP_Components[1]['kernel']
            LKL_all = kernel1.LKL(self.x_I)[4]
            LK_all = kernel1.LK(self.x_I)[2]
            KL_all = kernel1.KL(self.x_I)[2]
            self.LKL_V = LKL_all - LK_all @ self.GP_Models[1].K_inv @ KL_all + 1e-6 * torch.eye(2*self.n_I)
            self.LKL_V_inv = torch.linalg.inv(self.LKL_V)
            self.LKL_V_margin = kernel1.LKL(self.x_I)[1] - kernel1.LK(self.x_I)[1] @ self.GP_Models[1].K_inv @ kernel1.KL(self.x_I)[1] + 1e-6 * torch.eye(self.n_I)
            self.LKL_V_inv_margin = torch.linalg.inv(self.LKL_V_margin)
            return (u_KL, U, Lu_GP, Lu_GP_KL, GP_Trans_u_Lu)

    def Loss_for_Censored_Component(self, x):
        u_KL = self.u_KL_initial
        x = torch.tensor(x)
        d_theta = self.PDE_Model.para_theta.shape[0]
        theta = x[0:d_theta]
        u_KL_censored = x[d_theta:d_theta + u_KL.shape[0]]
        if self.pde_operator == 7 :
            u_KL[:,1] = u_KL_censored
            u_3 = self.GP_Trans_u_Lu[1]['Lu_mean_I']  + self.GP_Trans_u_Lu[1]['u_KL_to_Lu_GP'] @ u_KL[:,1]
            u_KL[:,3] = self.GP_Trans_u_Lu[3]['u_to_u_KL'] @ (u_3 - self.GP_Trans_u_Lu[3]['u_mean_I'])
            u = torch.empty(self.n_I,self.p).double()
            Lu_GP= torch.empty(self.n_I,self.p).double()
            for i in range(self.p):
                u[:,i] = self.GP_Trans_u_Lu[i]['u_mean_I'] + self.GP_Trans_u_Lu[i]['u_KL_to_u'] @ u_KL[:,i]
                Lu_GP[:,i] = self.GP_Trans_u_Lu[i]['Lu_mean_I']  + self.GP_Trans_u_Lu[i]['u_to_Lu_GP'] @ (u[:,i%2]-self.GP_Components[i%2]['mean'])
            Lu_PDE = self.PDE_Model.Source(self.x_I, u, theta)
            lkh = torch.zeros((1, 3))
            lkh[0,0] = -0.5 * torch.sum(torch.square(u_KL)) 
            Lu_Error = Lu_PDE - Lu_GP
            lkh[0,2] =  -0.5 * torch.cat((Lu_Error[:,0],Lu_Error[:,2])) @ self.LKL_U_inv @ torch.cat((Lu_Error[:,0],Lu_Error[:,2])).T /self.GP_Components[0]['outputscale']
            #lkh[0,2] =  lkh[0,2] -0.5 * torch.cat((Lu_Error[:,1],Lu_Error[:,3])) @ self.LKL_V_inv @ torch.cat((Lu_Error[:,1],Lu_Error[:,3])).T /self.GP_Components[1]['outputscale']
            #print(torch.mean(torch.square(self.V_test-u[:,1])))
        elif self.pde_operator == 8 :
            u_KL[:,0] = u_KL_censored
            u_2 = self.GP_Trans_u_Lu[0]['Lu_mean_I']  + self.GP_Trans_u_Lu[0]['u_KL_to_Lu_GP'] @ u_KL[:,0]
            u_KL[:,2] = self.GP_Trans_u_Lu[2]['u_to_u_KL'] @ (u_2 - self.GP_Trans_u_Lu[2]['u_mean_I'])

            u = torch.empty(self.n_I,self.p).double()
            Lu_GP= torch.empty(self.n_I,self.p).double()
            for i in range(self.p):
                u[:,i] = self.GP_Trans_u_Lu[i]['u_mean_I'] + self.GP_Trans_u_Lu[i]['u_KL_to_u'] @ u_KL[:,i]
                Lu_GP[:,i] = self.GP_Trans_u_Lu[i]['Lu_mean_I']  + self.GP_Trans_u_Lu[i]['u_to_Lu_GP'] @ (u[:,i%2]-self.GP_Components[i%2]['mean'])
            Lu_PDE = self.PDE_Model.Source(self.x_I, u, theta)
            lkh = torch.zeros((1, 3))
            lkh[0,0] = -0.5 * torch.sum(torch.square(u_KL)) 
            Lu_Error = Lu_PDE - Lu_GP
            #lkh[0,2] =  -0.5 * torch.cat((Lu_Error[:,0],Lu_Error[:,2])) @ self.LKL_U_inv @ torch.cat((Lu_Error[:,0],Lu_Error[:,2])).T /self.GP_Components[0]['outputscale']
            lkh[0,2] =  lkh[0,2] -0.5 * torch.cat((Lu_Error[:,1],Lu_Error[:,3])) @ self.LKL_V_inv @ torch.cat((Lu_Error[:,1],Lu_Error[:,3])).T /self.GP_Components[1]['outputscale']
            #print(torch.mean(torch.square(self.U_test-u[:,1])))
        return (torch.sum(lkh))

    def Loss_for_Censored_Component_adam(self, theta, u_KL_censored):
        u_KL = self.u_KL_initial
        d_theta = self.PDE_Model.para_theta.shape[0]
        if self.pde_operator == 7 :
            u_KL[:,1] = u_KL_censored
            u_3 = self.GP_Trans_u_Lu[1]['Lu_mean_I']  + self.GP_Trans_u_Lu[1]['u_KL_to_Lu_GP'] @ u_KL[:,1]
            u_KL[:,3] = self.GP_Trans_u_Lu[3]['u_to_u_KL'] @ (u_3 - self.GP_Trans_u_Lu[3]['u_mean_I'])
            u = torch.empty(self.n_I,self.p).double()
            Lu_GP= torch.empty(self.n_I,self.p).double()
            for i in range(self.p):
                u[:,i] = self.GP_Trans_u_Lu[i]['u_mean_I'] + self.GP_Trans_u_Lu[i]['u_KL_to_u'] @ u_KL[:,i]
                Lu_GP[:,i] = self.GP_Trans_u_Lu[i]['Lu_mean_I']  + self.GP_Trans_u_Lu[i]['u_to_Lu_GP'] @ (u[:,i%2]-self.GP_Components[i%2]['mean'])
            Lu_PDE = self.PDE_Model.Source(self.x_I, u, theta)
            lkh = torch.zeros((1, 3))
            lkh[0,0] = -0.5 * torch.sum(torch.square(u_KL)) 
            Lu_Error = Lu_PDE - Lu_GP
            lkh[0,2] =  -0.5 * torch.cat((Lu_Error[:,0],Lu_Error[:,2])) @ self.LKL_U_inv @ torch.cat((Lu_Error[:,0],Lu_Error[:,2])).T /self.GP_Components[0]['outputscale']
            #lkh[0,2] =  lkh[0,2] -0.5 * torch.cat((Lu_Error[:,1],Lu_Error[:,3])) @ self.LKL_V_inv @ torch.cat((Lu_Error[:,1],Lu_Error[:,3])).T /self.GP_Components[1]['outputscale']
            #print(torch.mean(torch.square(self.V_test-u[:,1])))
        elif self.pde_operator == 8 :
            u_KL[:,0] = u_KL_censored
            u_2 = self.GP_Trans_u_Lu[0]['Lu_mean_I']  + self.GP_Trans_u_Lu[0]['u_KL_to_Lu_GP'] @ u_KL[:,0]
            u_KL[:,2] = self.GP_Trans_u_Lu[2]['u_to_u_KL'] @ (u_2 - self.GP_Trans_u_Lu[2]['u_mean_I'])

            u = torch.empty(self.n_I,self.p).double()
            Lu_GP= torch.empty(self.n_I,self.p).double()
            for i in range(self.p):
                u[:,i] = self.GP_Trans_u_Lu[i]['u_mean_I'] + self.GP_Trans_u_Lu[i]['u_KL_to_u'] @ u_KL[:,i]
                Lu_GP[:,i] = self.GP_Trans_u_Lu[i]['Lu_mean_I']  + self.GP_Trans_u_Lu[i]['u_to_Lu_GP'] @ (u[:,i%2]-self.GP_Components[i%2]['mean'])
            Lu_PDE = self.PDE_Model.Source(self.x_I, u, theta)
            lkh = torch.zeros((1, 3))
            lkh[0,0] = -0.5 * torch.sum(torch.square(u_KL)) 
            Lu_Error = Lu_PDE - Lu_GP
            #lkh[0,2] =  -0.5 * torch.cat((Lu_Error[:,0],Lu_Error[:,2])) @ self.LKL_U_inv @ torch.cat((Lu_Error[:,0],Lu_Error[:,2])).T /self.GP_Components[0]['outputscale']
            lkh[0,2] =  lkh[0,2] -0.5 * torch.cat((Lu_Error[:,1],Lu_Error[:,3])) @ self.LKL_V_inv @ torch.cat((Lu_Error[:,1],Lu_Error[:,3])).T /self.GP_Components[1]['outputscale']
            #print(torch.mean(torch.square(self.U_test-u[:,1])))
        return (torch.sum(lkh))

