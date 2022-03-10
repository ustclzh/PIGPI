import numpy as np
import dill
import pickle
from scipy.io import savemat
import torch
import time
#import matplotlib.pyplot as plt
#import pyswarms
from scripts import MAGI_PDE_Infer # inferred module
from scripts import Generating_data # 
from scripts import Two_Stage_Method # 


def run_one_instance_map(operator, source_term, boundary_condition, noisy_known, sigma_e):
    Err_MAGI = torch.zeros(60,6)
    Err_TS = torch.zeros(60,6)
    p = 1*(source_term<=5) +2*(source_term>=6)
    Est_sigma = torch.zeros(60,6,p)
    RES_MAGI = []
    RES_TS = []
    p = 3 * (source_term ==1) + 1 * (source_term ==2) + 3 * (source_term ==3) + 3 * (source_term ==4) + 2 * (source_term ==5) + 3 * (source_term >=6) + 3 * (source_term ==0) 
    THETA_MAGI = torch.zeros(60,6,p)
    THETA_TS = torch.zeros(60,6,p)
    Err_MAGI_r = torch.zeros(60,6,p)
    Err_TS_r = torch.zeros(60,6,p)
    filename= 'res/all_eg_source'+str(source_term)+'_'+'b'+str(boundary_condition)+'err'+str(sigma_e)+'noise'+str(noisy_known)+'.mat'
    filename2= 'res/all_eg_source'+str(source_term)+'_'+'b'+str(boundary_condition)+'err'+str(sigma_e)+'noise'+str(noisy_known)+'.pkl'
    for num in range(60):
        #design_instance = num
        print(num)
        ind = 0
        for i in range(3):
            n_obs = 30* (2**i)
            for j in range(3-i):
                n_I = n_obs * (2**j)
                PDEmodel=Generating_data.Generating_data(
                    n_obs =n_obs, 
                    n_I = n_I, 
                    sigma_e=sigma_e, 
                    noisy_known = noisy_known, 
                    pde_operator = operator, 
                    source_term = source_term, 
                    boundary_condition=boundary_condition, 
                    design_instance = num
                    )
                magi= MAGI_PDE_Infer.MAGI_PDE_Infer(PDEmodel, KL=False) # call inference class
                magi.Sample_Using_HMC(n_epoch = 6000, lsteps=100, epsilon=5e-5, n_samples=20000, Normal_Approxi_Only = True)
                ts = Two_Stage_Method.Two_Stage_Method(PDEmodel) # call inference class
                ts.minimize_msl()
                theta_CI_NA = magi.Posterior_Summary()
                Err_MAGI[num,ind] = magi.Map_Estimation['theta_err']
                Err_TS[num,ind] = ts.Two_Stage_Estimation['theta_err']
                THETA_MAGI[num,ind] = magi.Map_Estimation['theta_MAP']
                THETA_TS[num,ind] = ts.Two_Stage_Estimation['theta_ts']
                Err_MAGI_r[num,ind] = magi.Map_Estimation['theta_err_relative']
                Err_TS_r[num,ind] = ts.Two_Stage_Estimation['theta_err_relative']
                Est_sigma[num,ind] = magi.Map_Estimation['sigma_e_sq_MAP']
                ind +=1
                RES_MAGI.append(magi.Map_Estimation)
                RES_TS.append(ts.Two_Stage_Estimation)
        ERR_summary = {
            'err_MAGI':Err_MAGI.numpy(),
            'err_TS': Err_TS.numpy(),
            'err_MAGI_r':Err_MAGI_r.numpy(),
            'err_TS_r': Err_TS_r.numpy(),
            'theta_true': PDEmodel.theta_true.numpy(),
            'sigma_e_map':Est_sigma.numpy(),
            'theta_magi':THETA_MAGI.numpy(),
            'theta_ts': THETA_TS.numpy()
        }
        f = open(filename2, 'wb')
        pickle.dump([RES_MAGI, RES_TS], f)
        f.close()
        savemat(filename,ERR_summary)
    return(ERR_summary,RES_MAGI, RES_TS)
