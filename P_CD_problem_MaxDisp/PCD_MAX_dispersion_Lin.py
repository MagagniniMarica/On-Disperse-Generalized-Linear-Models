# -*- coding: utf-8 -*-
"""

@author: Marica Magagnini

D-PDP-GLM problem implementation.
This problem is only for Linear case, which can be solved by gurobi 

"""

from pyomo import environ as pym
import numpy as np
import pandas as pd

def P_max_dispersion_Lin_(dispersion, dataset, target, features, M, 
                 B0, P, tau, theta, X0):
    
    #
    # Model definition
    #
    m = pym.ConcreteModel(name = 'D-PDP-{GLM} - Lin')
    

    #
    # Indexes
    #
    m.j = pym.Set(initialize = features)                     # All features
    features_bias = pd.Index(['bias']).append(features)
    m.j_b0 = pym.Set(initialize = features_bias)             # All features and bias
    
    
    # Number of instances
    N =  len(dataset)   
    m.n = pym.Set(initialize=dataset.index.tolist())
    
    # Number of known regressors
    Q = len(B0)
    m.q = pym.RangeSet(0,Q-1)
    
    #Number of new regressors
    m.p = pym.RangeSet(0,P-1)
    
    #
    # Parameters
    #
    
    # Target variable
    m.y = pym.Param(m.n, initialize=target)
    epsilon  = 1e-9
  
    
    # Dataset with exceeding column of 1s that refers to the bias term
    dataset_bias = dataset.copy()
    dataset_bias['bias'] = np.ones(len(dataset))
    def x_init(m,n,j):
        return dataset_bias.iloc[n][j]
    m.x = pym.Param(m.n,m.j_b0, initialize=x_init, mutable=True)
    
    
    # If the dispersion is on the response, we need to initialize  X^0 
    ############   
    if  dispersion == 'o1':
        m.X0  = pym.Param(m.j_b0, initialize=X0)
    ############ 
    
    
    
    # beta coefficients of known regressors 
    def B0_init(m,q,j):
        return B0.iloc[q][j]
    m.B0 = pym.Param(m.q,m.j_b0, initialize=B0_init,mutable=False)
    
    
    #
    # variables
    #
    m.beta = pym.Var(m.p, m.j_b0, within=pym.Reals, bounds= (-30,30))
    m.obj =  pym.Var(within=pym.NonNegativeReals) 

    
    if dispersion == 'l1':     
        m.e = pym.Var(m.q, m.p,m.j, within = pym.Binary)
        m.ze = pym.Var(m.q, m.p,m.j, within = pym.NonNegativeReals)
        m.t = pym.Var(m.p, m.p,m.j, within = pym.Binary)
        m.zt  = pym.Var(m.p, m.p,m.j, within = pym.NonNegativeReals)
    elif dispersion == 'dsa':
        m.r = pym.Var(m.p,m.p, m.j, within=pym.Binary)
        m.xi_r = pym.Var(m.p,m.j, within=pym.Binary) # \xi_p^+
        m.xi_l = pym.Var(m.p,m.j, within=pym.Binary) # \xi_p^-
    elif dispersion == 'o1':
        m.v = pym.Var(m.p,m.p, within = pym.Binary)
        m.w = pym.Var(m.p,m.q, within = pym.Binary)
        
    # sparsity
    m.xi = pym.Var(m.p,m.j, within=pym.Binary)

    # Objective function 
    #
    def objfunction_(m):
        return m.obj
    m.objfunction = pym.Objective(rule=objfunction_,  sense=pym.maximize)
    
    
    #
    # Initialize constraints
    #
    
    if dispersion == 'l1':
        
        def c_ed_1_(m,q,p):
            return pym.quicksum(m.ze[q,p,j] for j in m.j) >= m.obj
        
        def c_ed_2_(m,q,p,j):
            return m.ze[q,p,j] >= m.B0[q,j] - m.beta[p,j]
        def c_ed_3_(m,q,p,j):
            return m.ze[q,p,j] >= - (m.B0[q,j] - m.beta[p,j])
        
        def c_ed_4_(m,q,p,j):
            return m.ze[q,p,j] <= m.B0[q,j] - m.beta[p,j] + M*(1-m.e[q,p,j])
        def c_ed_5_(m,q,p,j):
            return m.ze[q,p,j] <= - (m.B0[q,j] - m.beta[p,j]) + M*m.e[q,p,j]
        
        
        def c_id_1_(m,p,p_prime):
            if p < p_prime:
                return pym.quicksum(m.zt[p,p_prime,j] for j in m.j) >= m.obj
            else: 
                return pym.Constraint.Skip
        
        def c_id_2_(m,p,p_prime,j):
            if p < p_prime:  
                return m.zt[p,p_prime,j] >= m.beta[p,j] - m.beta[p_prime,j]
            else: 
                return pym.Constraint.Skip
        def c_id_3_(m,p,p_prime,j):
            if p < p_prime:

                return m.zt[p,p_prime, j] >= - (m.beta[p,j] - m.beta[p_prime,j])
            else: 
                return pym.Constraint.Skip
        
        def c_id_4_(m,p,p_prime,j):
            if p < p_prime:
                return m.zt[p,p_prime,j] <= m.beta[p,j] - m.beta[p_prime,j] + M*(1-m.t[p,p_prime,j])
            else: 
                return pym.Constraint.Skip
        def c_id_5_(m,p,p_prime,j):
            if p < p_prime:
                return m.zt[p,p_prime, j] <= - (m.beta[p,j] - m.beta[p_prime,j]) + M*m.t[p,p_prime,j]
            else: 
                return pym.Constraint.Skip
        
        
        
    
    elif dispersion == 'l2':
       
        def c_extra_dispersion(m,q,p):
            return pym.quicksum((m.B0[q,j] - m.beta[p,j])**2 for j in m.j ) >= m.obj**2
        
        def c_intra_dispersion(m,p,p_prime):
            if p != p_prime:
                return pym.quicksum((m.beta[p,j] - m.beta[p_prime,j])**2 for j in m.j ) >= m.obj**2
            else: 
                return pym.Constraint.Skip
        

        
    elif dispersion =='dsa':
        def c_extra_dispersion(m,q,p):
            return pym.quicksum( 1- m.xi[p,j] for j in m.j  if m.B0[q,j] != 0  ) + pym.quicksum(m.xi[p,j]   for j in m.j  if m.B0[q,j] == 0) >= m.obj
        
        def c_intra_dispersion(m,p,p_prime):
            if p < p_prime:
                return pym.quicksum(m.xi[p,j] + m.xi[p_prime,j]-2*m.r[p,p_prime,j] for j in m.j) >= m.obj
            else: 
                return pym.Constraint.Skip
        
        def c_r1(m,p,p_prime,j):
            if p < p_prime:
                return m.r[p,p_prime,j] <= m.xi[p,j]
            else: 
                return pym.Constraint.Skip
        def c_r2(m,p,p_prime,j):
            if p < p_prime:
                return m.r[p,p_prime,j] <= m.xi[p_prime,j]
            else: 
                return pym.Constraint.Skip
        def c_r3(m,p,p_prime,j):
            if p < p_prime:
                return m.r[p,p_prime,j] >= m.xi[p,j] + m.xi[p_prime,j] -1
            else: 
                return pym.Constraint.Skip
        
        def cxi_(m,p,j):
          return m.xi[p,j] == m.xi_r[p,j] + m.xi_l[p,j]
        def cxi_lr_(m,p,j):
          return m.xi_r[p,j] + m.xi_l[p,j] <= 1
        def cxi_1_(m,p,j):
          return m.beta[p,j] >= epsilon - M*(1- m.xi_r[p,j]  )
        def cxi_2_(m,p,j):
          return m.beta[p,j] <= M*m.xi_r[p,j]
        def cxi_3_(m,p,j):
          return m.beta[p,j] <= -epsilon + M*(1- m.xi_l[p,j]  )
        def cxi_4_(m,p,j):
          return m.beta[p,j] >= -M*m.xi_l[p,j]
   
                       
    elif dispersion == 'o1':
        
        def c_intra_dispersion_1_(m, p, p_prime):
            if p < p_prime:
                return  m.v[p,p_prime]  + m.v[p_prime,p] == 1
            else: 
                return pym.Constraint.Skip
        def c_intra_dispersion_2_(m, p, p_prime, p_sec):
            if p != p_prime and p != p_sec and p_prime != p_sec:
                return m.v[p,p_sec] >= m.v[p,p_prime] + m.v[p_prime, p_sec] -1
            else:
                return pym.Constraint.Skip   
        def c_intra_dispersion_3_(m, p, p_prime):
            if p != p_prime:
                return pym.quicksum(m.beta[p,j]*m.X0[j]   for j in m.j_b0) -  pym.quicksum(m.beta[p_prime,j]*m.X0[j] for j in m.j_b0) + M*(1-m.v[p,p_prime]) >= m.obj
            else: 
                return pym.Constraint.Skip
        
        
        def c_extra_dispersion_1_(m, p,q,p_prime):
            if p != p_prime:
                return   m.v[p,p_prime] >= m.w[p,q] -m.w[p_prime,q]
            else:
                return pym.Constraint.Skip
        def c_extra_dispersion_2_(m, p,q):
            return  pym.quicksum(m.beta[p,j]*m.X0[j] for j in m.j_b0)- pym.quicksum(m.B0[q,j]*m.X0[j]   for j in m.j_b0) + M*(1-m.w[p,q]) >= m.obj
        def c_extra_dispersion_3_(m, p,q):
            return  pym.quicksum(m.B0[q,j]*m.X0[j]   for j in m.j_b0) -  pym.quicksum(m.beta[p,j]*m.X0[j] for j in m.j_b0) + M*m.w[p,q] >= m.obj
             
       
        
    
    # Accuracy
    def beta_xn_(m,n,p):
        return pym.quicksum(m.beta[p, j] * m.x[n, j] for j in m.j_b0)
        
    def c_accuracy_p_(m, p):
        return sum( (m.y[n] - beta_xn_(m,n,p) )**2 for n in m.n) <= N*(tau)

   
    
    
    # Sparsity 
    def c_sparsity_1_(m,p):
        return pym.quicksum(m.xi[p,j] for j in m.j) <= theta 
    def c_sparsity_2a_(m,p,j):
        return -M*m.xi[p,j] <= m.beta[p,j]
    def c_sparsity_2b_(m,p,j):
        return  m.beta[p,j] <= M*m.xi[p,j]
            
    #
    # Declare constraints
    #
    
    if dispersion == 'l1':
        m.c_Edis1 = pym.Constraint(m.q, m.p, rule = c_ed_1_)
        m.c_Edis2 = pym.Constraint(m.q, m.p, m.j, rule = c_ed_2_)
        m.c_Edis3 = pym.Constraint(m.q, m.p, m.j, rule = c_ed_3_)
        m.c_Edis4 = pym.Constraint(m.q, m.p, m.j, rule = c_ed_4_)
        m.c_Edis5 = pym.Constraint(m.q, m.p, m.j, rule = c_ed_5_)
    
        m.c_Idis1 = pym.Constraint(m.p, m.p, rule = c_id_1_)
        m.c_Idis2 = pym.Constraint(m.p, m.p, m.j, rule = c_id_2_)
        m.c_Idis3 = pym.Constraint(m.p, m.p, m.j, rule = c_id_3_)
        m.c_Idis4 = pym.Constraint(m.p, m.p, m.j, rule = c_id_4_)
        m.c_Idis5 = pym.Constraint(m.p, m.p, m.j, rule = c_id_5_)
    
    elif dispersion == 'l2':
        m.c_Edis = pym.Constraint(m.q, m.p, rule = c_extra_dispersion)
        m.c_Idis = pym.Constraint(m.p, m.p, rule = c_intra_dispersion)

    elif dispersion == 'dsa':
        m.c_Edis = pym.Constraint(m.q, m.p, rule = c_extra_dispersion)
        m.c_Idis = pym.Constraint(m.p, m.p, rule = c_intra_dispersion)
        m.c_Idis_r1 = pym.Constraint(m.p, m.p, m.j, rule = c_r1)
        m.c_Idis_r2 = pym.Constraint(m.p, m.p, m.j, rule = c_r2)
        m.c_Idis_r3 = pym.Constraint(m.p, m.p, m.j, rule = c_r3)
        m.c_xi_ = pym.Constraint(m.p,m.j, rule = cxi_)
        m.c_xi_lr_ = pym.Constraint(m.p,m.j, rule = cxi_lr_)
        m.c_xi_1_ = pym.Constraint(m.p,m.j, rule = cxi_1_)
        m.c_xi_2_ = pym.Constraint(m.p,m.j, rule = cxi_2_)
        m.c_xi_3_ = pym.Constraint(m.p,m.j, rule = cxi_3_)
        m.c_xi_4_ = pym.Constraint(m.p,m.j, rule = cxi_4_)
    
    elif dispersion  == 'o1': 
   
        m.c_Edis1 = pym.Constraint(m.p, m.q, m.p, rule = c_extra_dispersion_1_)
        m.c_Edis2 = pym.Constraint(m.p, m.q, rule = c_extra_dispersion_2_)
        m.c_Edis3 = pym.Constraint(m.p, m.q, rule = c_extra_dispersion_3_)
    
        m.c_Idis1 = pym.Constraint(m.p, m.p, rule = c_intra_dispersion_1_)
        m.c_Idis2 = pym.Constraint(m.p,m.p, m.p, rule = c_intra_dispersion_2_)
        m.c_Idis3 = pym.Constraint(m.p, m.p, rule = c_intra_dispersion_3_)
   
    
    # Accuracy
    m.c_acc_p_ = pym.Constraint(m.p, rule = c_accuracy_p_)
    
    # Sparsity
    m.c_s1 = pym.Constraint(m.p, rule= c_sparsity_1_)
    if dispersion != 'dsa':
        m.c_s2a = pym.Constraint(m.p, m.j, rule= c_sparsity_2a_)
        m.c_s2b = pym.Constraint(m.p, m.j, rule= c_sparsity_2b_)
        
    
    return m 
