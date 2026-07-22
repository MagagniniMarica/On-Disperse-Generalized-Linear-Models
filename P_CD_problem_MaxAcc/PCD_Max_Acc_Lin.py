
"""
@author: Marica Magagnini
"""


from pyomo import environ as pym
import numpy as np
import pandas as pd




# dispersion  = 'l1','l2','dsa','o1' 
# model = regression, logisticRegression, ..  !!Affects the accuracy constraint and the obj!!
def P_cond_disp_(dispersion, dataset, target, features, 
                          betaS, P, gamma, tau, theta,  M, X0):
    
    #
    # Model definition
    #
    m = pym.ConcreteModel(name = 'P-CD Problem - MAX Accuracy - LinP-Conditional Dispersion Problem - Lin optimal')
    
    #
    # Indexes
    #
    m.j = pym.Set(initialize = features)                     # All features
    features_bias = pd.Index(['bias']).append(features)
    m.j_b0 = pym.Set(initialize = features_bias)             # All features and bias
    

    # Number of instances
    N =  len(dataset)   
    m.n = pym.RangeSet(0,N-1)
    
    # Number of known regressors
    Q = len(betaS)
    m.q = pym.RangeSet(0,Q-1)
    
    #Number of new regressors
    m.p = pym.RangeSet(0,P-1)
    
    #
    # Parameters
    #
    
    # Target variable
    m.y = pym.Param(m.n, initialize=target)
    
    
    dataset_bias = dataset.copy()
    dataset_bias['bias'] = 1.0
    x_dict = {(n, j): dataset_bias.iloc[n, dataset_bias.columns.get_loc(j)]
          for n in range(len(dataset_bias))
          for j in dataset_bias.columns}
    m.x = pym.Param(m.n, m.j_b0, initialize=x_dict, mutable=True, within = pym.Reals)
   
    
    # If the dispersion is on the output, we need to initialize the X^0 set
    ############   
    if  dispersion == 'o1':
        if type(X0) == pd.core.series.Series:
            len_X0 = 1
        else:
            len_X0 = len(X0)
        if len_X0 == 1:
            m.X0  = pym.Param(m.j_b0, initialize=X0)
        else:
            m.n0 = pym.RangeSet(0,len_X0-1)
            def X0_init(m,n0,j):
                return X0.iloc[n0][j]
            m.X0 = pym.Param(m.n0,m.j_b0, initialize=X0_init, mutable=True)
    ############   
    
    
    
    # beta coefficients of known regressors 
    def betaS_init(m,q,j):
        return betaS.iloc[q][j]
    m.betaS = pym.Param(m.q,m.j_b0, initialize=betaS_init,mutable=False)
    
    
    #
    # variables
    #
    ulb = (-30,30) if len(features) <20 else (-50,50) if  (len(features) > 20 and len(features) < 70) else (-1e5, 1e5)
    m.beta = pym.Var(m.p, m.j_b0, within=pym.Reals, bounds=ulb) #(-30,30) BH, (-50,50) ITT, (-1e5, 1e5) CC

    if dispersion == 'l1':     
        m.e = pym.Var(m.q, m.p,m.j, within = pym.Binary)
        m.ze = pym.Var(m.q, m.p,m.j, within = pym.NonNegativeReals)
        m.t = pym.Var(m.p, m.p,m.j, within = pym.Binary)
        m.zt  = pym.Var(m.p, m.p,m.j, within = pym.NonNegativeReals)
    elif dispersion == 'dsa':
        m.z = pym.Var(m.p,m.p, m.j, within=pym.Binary)
    elif dispersion == 'o1':
        m.v = pym.Var(m.p,m.p, within = pym.Binary)
        m.w = pym.Var(m.p,m.q, within = pym.Binary)
        
        
    # sparsity
    m.xi = pym.Var(m.p,m.j, within=pym.Binary)
    
    #
    # Objective function 
    #
    
    def beta_xn_(m, n,p):
        return pym.quicksum(m.beta[p, j] * m.x[n, j] for j in m.j_b0)
    
    def objfunction_(m):
        return   pym.quicksum( (1/N) *pym.quicksum((m.y[n] - beta_xn_(m, n,p))**2 for n in m.n )for p in m.p)
        
      


         
    m.objfunction = pym.Objective(rule=objfunction_,  sense=pym.minimize)
    
     
    
    #
    # Initialize constraints
    #
    
    if dispersion == 'l1':
        
        def c_ed_1_(m,q,p):
            return pym.quicksum(m.ze[q,p,j] for j in m.j) >= gamma
        
        def c_ed_2_(m,q,p,j):
            return m.ze[q,p,j] >= m.betaS[q,j] - m.beta[p,j]
        def c_ed_3_(m,q,p,j):
            return m.ze[q,p,j] >= - (m.betaS[q,j] - m.beta[p,j])
        
        def c_ed_4_(m,q,p,j):
            return m.ze[q,p,j] <= m.betaS[q,j] - m.beta[p,j] + M*(1-m.e[q,p,j])
        def c_ed_5_(m,q,p,j):
            return m.ze[q,p,j] <= - (m.betaS[q,j] - m.beta[p,j]) + M*m.e[q,p,j]
        
        
        def c_id_1_(m,p,p_prime):
            if p < p_prime:
                return pym.quicksum(m.zt[p,p_prime,j] for j in m.j) >= gamma
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
            return pym.quicksum((m.betaS[q,j] - m.beta[p,j])**2 for j in m.j ) >= gamma**2
        
        def c_intra_dispersion(m,p,p_prime):
            if p < p_prime:
                return pym.quicksum((m.beta[p,j] - m.beta[p_prime,j])**2 for j in m.j ) >= gamma**2
            else: 
                return pym.Constraint.Skip
            
    elif dispersion =='dsa':
        def c_extra_dispersion(m,q,p):
            return pym.quicksum( 1- m.xi[p,j] if m.betaS[q,j] != 0 else 0 for j in m.j  ) + pym.quicksum(m.xi[p,j] if m.betaS[q,j] == 0 else 0 for j in m.j  ) >= gamma
        
        def c_intra_dispersion(m,p,p_prime):
            if p < p_prime:
                return pym.quicksum(m.xi[p,j] + m.xi[p_prime,j]-2*m.z[p,p_prime,j] for j in m.j) >= gamma
            else: 
                return pym.Constraint.Skip
        
        def c_z1(m,p,p_prime,j):
            if p < p_prime:
                return m.z[p,p_prime,j] <= m.xi[p,j]
            else: 
                return pym.Constraint.Skip
        def c_z2(m,p,p_prime,j):
            if p < p_prime:
                return m.z[p,p_prime,j] <= m.xi[p_prime,j]
            else: 
                return pym.Constraint.Skip
        def c_z3(m,p,p_prime,j):
            if p < p_prime:
                return m.z[p,p_prime,j] >= m.xi[p,j] + m.xi[p_prime,j] -1
            else: 
                return pym.Constraint.Skip
            
                      
    elif dispersion == 'o1':
        if len_X0 == 1:
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
                    return pym.quicksum(m.beta[p,j]*m.X0[j]   for j in m.j_b0) -  pym.quicksum(m.beta[p_prime,j]*m.X0[j] for j in m.j_b0) + M*(1-m.v[p,p_prime]) >= gamma
                else: 
                    return pym.Constraint.Skip
            
            
            def c_extra_dispersion_1_(m, p,q,p_prime):
                if p != p_prime:
                    return   m.v[p,p_prime] >= m.w[p,q] -m.w[p_prime,q]
                else:
                    return pym.Constraint.Skip
            def c_extra_dispersion_2_(m, p,q):
                return  pym.quicksum(m.beta[p,j]*m.X0[j] for j in m.j_b0)- pym.quicksum(m.betaS[q,j]*m.X0[j]   for j in m.j_b0) + M*(1-m.w[p,q]) >= gamma
            def c_extra_dispersion_3_(m, p,q):
                return  pym.quicksum(m.betaS[q,j]*m.X0[j]   for j in m.j_b0) -  pym.quicksum(m.beta[p,j]*m.X0[j] for j in m.j_b0) + M*m.w[p,q] >= gamma
                 
            
        else:
            print('Not implemented.')
        
    
    
    
                    
    else:
        print('Dispersion norm not recognized. ')
        
        
        
    def c_accuracy_p_(m, p):
        return pym.quicksum( (m.y[n] - beta_xn_(m, n,p) )**2 for n in m.n) <= N*(tau)
            
       
        
        
    # Sparsity 
    def c_sparsity_1_(m,p):
        return pym.quicksum(m.xi[p,j] for j in m.j) <= theta   
    m.c_s1 = pym.Constraint(m.p, rule= c_sparsity_1_)
       
    if dispersion == 'dsa':
            m.xip = pym.Var(m.p,m.j, within=pym.Binary)
            m.xim = pym.Var(m.p,m.j, within=pym.Binary)
            def cs_pm_(m,p,j):
                return m.xip[p,j] +m.xim[p,j] <= 1
            m.cspm = pym.Constraint(m.p, m.j, rule= cs_pm_)
            def cs_p1_(m,p,j):
                return m.beta[p,j] >= 10e-5 - M * (1- m.xip[p,j])
            m.csp1 = pym.Constraint(m.p, m.j, rule= cs_p1_)
            def cs_p2_(m,p,j):
                return m.beta[p,j] <=  M *  m.xip[p,j]
            m.csp2 = pym.Constraint(m.p, m.j, rule= cs_p2_)
            def cs_m1_(m,p,j):
                return m.beta[p,j] <= -10e-5 + M * (1- m.xim[p,j])
            m.csm1 = pym.Constraint(m.p, m.j, rule= cs_m1_)
            def cs_m2_(m,p,j):
                return m.beta[p,j] >=  -M *  m.xim[p,j]
            m.csm2 = pym.Constraint(m.p, m.j, rule= cs_m2_)
            
            def cxi_(m,p,j):
                return m.xi[p,j] == m.xip[p,j] +m.xim[p,j] 
            m.cxi = pym.Constraint(m.p, m.j, rule= cxi_)
            
    else:
        def c_sparsity_2a_(m,p,j):
            return -M*m.xi[p,j] <= m.beta[p,j]
        def c_sparsity_2b_(m,p,j):
            return  m.beta[p,j] <= M*m.xi[p,j] 
        m.c_s2a = pym.Constraint(m.p, m.j, rule= c_sparsity_2a_)
        m.c_s2b = pym.Constraint(m.p, m.j, rule= c_sparsity_2b_)    
        
            
      
    
    
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
        m.c_Idis_z1 = pym.Constraint(m.p, m.p, m.j, rule = c_z1)
        m.c_Idis_z2 = pym.Constraint(m.p, m.p, m.j, rule = c_z2)
        m.c_Idis_z3 = pym.Constraint(m.p, m.p, m.j, rule = c_z3)
    
    
    elif dispersion  == 'o1': 
        if len_X0 == 1: 
            m.c_Edis1 = pym.Constraint(m.p, m.q, m.p, rule = c_extra_dispersion_1_)
            m.c_Edis2 = pym.Constraint(m.p, m.q, rule = c_extra_dispersion_2_)
            m.c_Edis3 = pym.Constraint(m.p, m.q, rule = c_extra_dispersion_3_)
        
            m.c_Idis1 = pym.Constraint(m.p, m.p, rule = c_intra_dispersion_1_)
            m.c_Idis2 = pym.Constraint(m.p,m.p, m.p, rule = c_intra_dispersion_2_)
            m.c_Idis3 = pym.Constraint(m.p, m.p, rule = c_intra_dispersion_3_)
        else:
            print('Not implemented. ')
        
               
    
    m.c_acc_p_ = pym.Constraint(m.p, rule = c_accuracy_p_)
   
    
    
    return m

