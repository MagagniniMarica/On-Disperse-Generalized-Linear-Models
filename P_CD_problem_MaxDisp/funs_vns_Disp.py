# -*- coding: utf-8 -*-
"""
@author: Marica Magagnini

This file contains all the support functions for the VNS strategy apllied to the 
maximum dipersion problem. 

Note for paper reader: 
- SQ = \B_0 (known models)
- SP = \B (models to construct)
- FSP = \Xi (set of P binary vectors)
- fsp  = \xi (binary vector)
"""

import numpy as np
import random
from pyomo import environ as pym
import pandas as pd
import itertools
from f_print import get_obj_sol_

# This function update the VNS index k 
def update_k_(k,K):
    if k < K-1:
        return k+1
    else:
        return 1


# 'Classical Hamming distance' between two feature selection structures 
def count_differences(fsp1, fsp2):
    sorted_fsp1 = sorted(fsp1)
    sorted_fsp2 = sorted(fsp2)

    count = 0
    for row1, row2 in zip(sorted_fsp1, sorted_fsp2):
        count += sum(b1 != b2 for b1, b2 in zip(row1, row2))
    return count

#
# Control that the feature selected can reach the accuracy requirements.
# If not, build another one. If yes, hold the solution cofficients 
# as feasible solution of the maximal dispersion problem
#
def acc_requirements(GLM, dataset, target, features, FSP_p, J1, tau):
    FSP_p_loss, FSP_p_coeff = max_acc_p_(GLM, dataset, target, features, FSP_p, J1)
    if FSP_p_loss <= tau:
        return True, FSP_p_coeff
    else:
        print('Combinatorial attemp not satisfying accuracy requirements.')
        return False, []

# Given a set of P models which satisfy the accuracy constraints, this function reconstructs 
# its combinatorial structure with respect to 'l1' coeffients dispersion. See Algorithm 4, (d1, alpha =1)
# in the Supplementary Material
def extract_l1_(P,FSP_maxAcc_coeff,SQ, features):
    #Inner dispersion structure
    E ={f: np.zeros((P,P)) for f in features}
    
    SP_ = pd.DataFrame(FSP_maxAcc_coeff)
    SP_ord = {}
    for f in features:
        s = SP_[f].sort_values(ascending=True)
        SP_ord[f] = s.reset_index()  # l’indice originale va in una colonna
        SP_ord[f].columns = ["P", "value"]
    
    
    for f in features:
        for r in range(P):
            for c in range(P):
                a = SP_ord[f].index[SP_ord[f]["P"] == r][0]
                a_val = SP_ord[f]['value'].iloc[r]
                b = SP_ord[f].index[SP_ord[f]["P"] == c][0]
                b_val = SP_ord[f]['value'].iloc[c]
                if a < b and not (a_val == 0 and b_val==0 ):
                    E[f][r,c]=1
                elif b < a and not (a_val == 0 and b_val==0 ):
                    E[f][r,c]= -1 
    
    #Inner + Outer Dispersion structure
    
    SP_SQ_ord = {}
    SP_SQ = pd.concat([SP_, SQ], ignore_index=True)
    
    Q = len(SQ)

    for f in features:
        s = SP_SQ[f].sort_values(ascending=True)
        SP_SQ_ord[f] = s.reset_index()  
        SP_SQ_ord[f].columns = ["PQ", "value"]

   
    # Costruiamo T
    T ={f: np.zeros((P,Q)) for f in features}
   
    for f in features:
        for p in range(P):
            for q in range(P, P+Q):
                a = SP_SQ_ord[f].index[SP_SQ_ord[f]["PQ"] == p][0]
                a_val = SP_SQ_ord[f]['value'].iloc[p]
                b = SP_SQ_ord[f].index[SP_SQ_ord[f]["PQ"] == q][0]
                b_val = SP_SQ_ord[f]['value'].iloc[q]
                
                
                if a < b and SP_[f].iloc[p] != 0 :
                    T[f][p,q-P] = 1
                    # i.e. \beta[p,f] <= \B0[q,f]
                elif a > b and SP_[f].iloc[p] != 0 :
                    T[f][p,q-P] = -1
                    #\beta[p,f] >= \B0[q,f]
    
    return E,T

# Given a set of P models which satisfy the accuracy constraints, this function reconstructs 
# its combinatorial structure with respect to 'o1' pedictors dispersion. See Algorithm 4, (d2, alpha =1)
# in the Supplementary Material
def extract_o1_(X0, SQ, P,FSP_maxAcc_coeff ):
    
    def betaTx0( beta, X0):
        return sum(beta[f]*X0[f] for f in X0.keys())
    # INNER DISPERSION v[i,j] = 1 if beta_i >= beta_j + gamma i,j = 1..P i!=j 
    
    SP_ = {p : betaTx0( FSP_maxAcc_coeff[p], X0) for p in range(P)}
    SP_ord = sorted(SP_, key=SP_.get, reverse=True) # decreasing order

    V = np.zeros((P,P))
    for r in range(P):
        for c in range(P):
            if SP_ord.index(r) < SP_ord.index(c):
                V[r,c]=1
    
    # OUTER DISPERSION w[i,j] = 1 if beta_i >= beta_j + gamma i = 1,..P, j=1, .., Q
    Q = len(SQ)
    SP_SQ = SP_

    if Q>1:
        for q in range(Q):
            beta = SQ.iloc[q]
            SP_SQ[P+q] = np.dot(beta,X0)
    else:
        SP_SQ[P] = np.dot(SQ,X0)
    
    SP_SQ_ord = sorted(SP_SQ, key=SP_SQ.get, reverse=True) # decreasing order
   
    
    W = np.zeros((P,Q))
    for r in range(P):
        for c in range(P,P+Q):
            if SP_SQ_ord.index(r) < SP_SQ_ord.index(c):
                W[r, c-P] = 1
    return V,W
   
# classical hamming dispersion  computation
def min_hamming_distance(FSP, SQ):
    
    hamming_distance = 1000
    # inner dsa dispersion
    for r1_idx, r1 in enumerate(FSP):
        for r2 in FSP[r1_idx+1:]:
            hd_r12 = sum(b1 != b2 for b1, b2 in zip(list(r1), list(r2)))
            if hd_r12 < hamming_distance:
                hamming_distance = hd_r12
                
    # outer dsa dispersion 

    eps = 1e-9
    FSQ = [[0 if abs(x) < eps else 1 for x in SQ.iloc[q]] for q in range(len(SQ))]
    for p in FSP:
        for q in FSQ:
            hd_pq = sum(b1 != b2 for b1, b2 in zip(list(p), list(q)))
            if hd_pq < hamming_distance:
                hamming_distance = hd_pq
                
    return hamming_distance


###############################################################################
# Perturbations
###############################################################################
# Perturbation 'l1','l2','o1'
def perturbation_(FSP_store, FSP, theta, k, P,
                  GLM, dataset, target, features, tau, J1,
                  dispersion, SQ, X0, solver_Iterations, trash_valid_fsp):
    def generate_valid_list(J1, theta, ref_list, n_flip):
        
        max_n_flip = 1
        if ref_list != [] :
            # Copy original list
            lst = ref_list.copy()

            #Indices
            indices = list(range(1, len(lst)))

            # Split 0 and 1 indices
            zeros = [i for i in indices if lst[i] == 0]
            ones = [i for i in indices if lst[i] == 1]
            
            # Maximum feasible exchange
            max_n_flip = min(len(zeros), len(ones))
            
        if n_flip < max_n_flip : 
                # Random choice
                chosen_zeros = random.sample(zeros, n_flip)
                chosen_ones = random.sample(ones, n_flip)

                # Exchange
                for i0, i1 in zip(chosen_zeros, chosen_ones):
                    lst[i0], lst[i1] = lst[i1], lst[i0]
        
        if ref_list == [] or  n_flip >= max_n_flip: 
            lst = [0] * J1
            lst[0] = 1  # Bias term 1
            sparsity =  random.randint(6, theta) ## In general random.randint(1, theta)
            indices = random.sample(range(1, J1), sparsity)
            for idx in indices:
                lst[idx] = 1

        return lst

    #
    # Returns the combinatorial stucture according to the dispersion
    # Extract from the set of max acc solutions
    #
    def C_(P,FSP_MaxACC_coeff,SQ, features, X0):
        V,W,E,T = (None, None,None,None)        
        if  dispersion == 'l1':
            E,T = extract_l1_(P,FSP_MaxACC_coeff,SQ, features)
        elif dispersion == 'o1':
            V,W = extract_o1_(X0, SQ, P,FSP_MaxACC_coeff ) 
        return  V,W,E,T
    

    (V,W,E,T) = (None,None,None,None)
    
    #
    # Comination of feasible availabe structures not already considered
    #
    combo_unique_tuples = list(itertools.combinations_with_replacement(trash_valid_fsp['valid'], P))
    for FSP_star in combo_unique_tuples:
        # VNS perturbation -> FSP_star has k elements different from FSP
        if (count_differences(FSP, FSP_star) >= k or FSP== [] ) and FSP_star not in FSP_store:
            FSP_store.add(FSP_star)
            FSP_maxAcc_Coeff = []
            for FSP_star_p in  FSP_star:
                FSP_star_p_loss, FSP_star_p_coeff = acc_requirements(GLM, dataset, target, features, FSP_star_p, J1, tau)
                FSP_maxAcc_Coeff.append(FSP_star_p_coeff)
            
            (V,W,E,T) = C_(P,FSP_maxAcc_Coeff,SQ, features, X0)
            return FSP_star, FSP_maxAcc_Coeff, V,W,E,T, trash_valid_fsp,FSP_store
            
            
    

    #
    # New construction
    #
    
    ref_list = []
    # Choose a reference structure
    if trash_valid_fsp['valid'] :
        ref_list = list(list(trash_valid_fsp['valid'])[-1])

    
        
    n_dst = 0       # number of distinct structure built
     
    # Generation phase
    FSPp = generate_valid_list(J1, theta, ref_list, k)  #n_flip=K
    
    # Check if FSPp is already known
    if tuple(FSPp) not in trash_valid_fsp['trash'] and tuple(FSPp) not in trash_valid_fsp['valid'] :
        acc_p, FSPp_MaxACC_coeff= acc_requirements(GLM, dataset, target, features, FSPp, J1, tau)
    else: 
        acc_p =False
          
        
    # Check accuracy constraints: if satisfied ---> allocate to valid structures
    if  acc_p:                   
        n_dst+=1
        trash_valid_fsp['valid'].add(tuple(FSPp))
        print('Found!')
        
        # Generate all the combinations using the incremented set of valid structures
        combo_dst = list(itertools.combinations_with_replacement(trash_valid_fsp['valid'], P))

        for FSP_star in combo_dst:
            if FSP_star not in FSP_store and (FSP == [] or count_differences(FSP, FSP_star) >= k):
                FSP_store.add(FSP_star)
                FSP_maxAcc_Coeff = []
                for FSP_star_p in  FSP_star:
                    FSP_star_p_loss, FSP_star_p_coeff = acc_requirements(GLM, dataset, target, features, FSP_star_p, J1, tau)
                    FSP_maxAcc_Coeff.append(FSP_star_p_coeff)
                (V,W,E,T) = C_(P,FSP_maxAcc_Coeff,SQ, features, X0)
                return FSP_star, FSP_maxAcc_Coeff, V,W,E,T, trash_valid_fsp, FSP_store
                

    #If FSPp is known or it does not satisfy the accuracy constraints ---> trash
    else: 
        trash_valid_fsp['trash'].add(tuple(FSPp))
 
                    

    return None,None,V,W,E,T, trash_valid_fsp, FSP_store



# Perturbation per dsa
def perturbation_dsa_(GLM, dataset, target, features,P,J1,k,tau, theta, gamma,SQ,FSP,FSP_store,trash_valid_fsp ):
    
    def generate_valid_list(J1, theta):  
        # Random construction
        lst = [0] * J1
        lst[0] = 1  # il primo elemento è sempre 1
        sparsity =  random.randint(5, theta) #OCCHIO 6 ce lo metto io perché senno non mi viene mai la minima accuratezza per come ho scelto i parametri
        indices = random.sample(range(1, J1), sparsity)
        for idx in indices:
            lst[idx] = 1
        return lst
    
    

    # for dsa dispersion 
    def hamming_distance(list1, list2):
        return sum(b1 != b2 for b1, b2 in zip(list1, list2))    
    
    # for dsa dispersion 
    def satisfies_gamma_constraint(fsp_star, fsq, gamma):
        # outer dsa dispersion
        for star_row in fsp_star:
            for fsq_row in fsq:
                if hamming_distance(star_row, fsq_row) < gamma:
                    h = hamming_distance(star_row, fsq_row)
                    print(f'FSQ_{fsq_row} , FSP_star_{star_row} --> hd : {h}')
                    return False
        # inner dsa dispersion
        for r1_idx, r1 in enumerate(fsp_star):
            for r2 in fsp_star[r1_idx+1:]:
                if hamming_distance(r1, r2) < gamma:
                    return False
        return True
    
    
    FSP_star = []
    FSP_maxAcc_Coeff = []
    
    eps = 1e-8
    FSQ = [[0 if abs(x) < eps else 1 for x in SQ.iloc[q]] for q in range(len(SQ))]
    
    
    #
    # Comination of feasible availabe structures not already considered
    #
    combo_unique_tuples = list(itertools.combinations_with_replacement(trash_valid_fsp['valid'], P))
    for FSP_star in combo_unique_tuples:
        # VNS perturbation -> FSP_star has k elements different from FSP
        if (FSP == [] or count_differences(FSP, FSP_star) >= k):
            if FSP_star not in FSP_store and satisfies_gamma_constraint(FSP_star, FSQ, gamma) :
                FSP_store.add(FSP_star)
                for p in range(P):
                    FSPp_acc, FSPp_coeff = max_acc_p_(GLM, dataset, target, features, FSP_star[p], J1)
                    FSP_maxAcc_Coeff.append(FSPp_coeff)
        
                return FSP_star, FSP_maxAcc_Coeff, trash_valid_fsp, FSP_store


    #
    # New construction
    #
    n_dst = 0       #  number of distinct structures built
    dst = list(trash_valid_fsp['valid'])        # list 
    dst_coeff= [max_acc_p_(GLM, dataset, target, features, fsp, J1)[1] for fsp in dst]   # coefficients
    
    max_attempts = 40
    attempts = 0
    
    #Generate new structures till I find:
        # - a new one
        # - with tau accuracy
        # - n_dst>=P
    while attempts < max_attempts :
        attempts +=1
        print(f'\n-------------\n attempt : {attempts} \n-------------\n')
        
        fsp_ = generate_valid_list(J1, theta)
        
        # Check accuracy if fsp_ not already known
        if tuple(fsp_) not in trash_valid_fsp['valid'] and tuple(fsp_) not in trash_valid_fsp['trash'] :
            fsp_acc, fsp_coeff = acc_requirements(GLM, dataset, target, features, fsp_, J1, tau)

        else:
            fsp_acc = False
            
            
        # Accuracy satisfied --> allocate to valid
        if fsp_acc :
            n_dst+=1
            dst.append(tuple(fsp_))
            trash_valid_fsp['valid'].add(tuple(fsp_))
            dst_coeff.append(fsp_coeff)
                
            #sort
            paired = list(zip(dst, dst_coeff))
                
           
            paired_sorted = sorted(paired, key=lambda x: x[0])
                
           
            dst_sorted, dst_coefff_sorted = zip(*paired_sorted)
            dst = list(dst_sorted)
            dst_coeff = list(dst_coefff_sorted)
        # Accuracy not satisfied --> allocate to trash
        else:
            trash_valid_fsp['trash'].add(tuple(fsp_))
        
        
        # Stop generating if there are at least P structures
        if len(dst)>= P:
            
            # Generate all possible combinations of P stuctures
            combo_dst = list(itertools.combinations(dst, P))
            combo_dst_coeff = list(itertools.combinations(dst_coeff, P))
            
            for FSP_star, FSP_maxAcc_Coeff in zip(combo_dst, combo_dst_coeff):
                attempts +=1
                if FSP_star not in FSP_store  and (FSP == [] or count_differences(FSP, FSP_star) >= k) and (satisfies_gamma_constraint(FSP_star, FSQ, gamma)):
                    FSP_store.add(FSP_star)
                    return FSP_star, FSP_maxAcc_Coeff, trash_valid_fsp, FSP_store
                
            
    
    return None, None, trash_valid_fsp, FSP_store

def abs_betaj_epsilon(B, epsilon):
    """
    This function check if all non zero features have absolute value greater than epsilon
    """
    for p, beta_dict in B.items():
        for f, value in beta_dict.items():
            # |beta_j| < epsilon
            if abs(value) < epsilon:
                return False
    return True
###############################################################################
# Accuracy check function
###############################################################################
def max_acc_p_(GLM, dataset, target, features, FSPp, J1):
    #
    # Model definition
    #
    m = pym.ConcreteModel(name = 'Min loss')
    
    #
    # Indexes
    #
    m.j = pym.Set(initialize = features)                     # All features
    features_bias = pd.Index(['bias']).append(features)
    m.j_b0 = pym.Set(initialize = features_bias)             # All features and bias
    
    
    # Number of instances
    N =  len(dataset)   
    m.n = pym.RangeSet(0,N-1)
    
    # Target variable
    m.y = pym.Param(m.n, initialize=target)
    
    # Dataset with exceeding column of 1s that refers to the bias term
    dataset_bias = dataset.copy()
    dataset_bias['bias'] = np.ones(len(dataset))
    def x_init(m,n,j):
        return dataset_bias.iloc[n][j]
    m.x = pym.Param(m.n,m.j_b0, initialize=x_init, mutable=True)
    
    #
    # variables
    #
    m.beta = pym.Var(m.j_b0, within=pym.Reals, bounds= (-30,30))
    
    # Fix to zero some variables as a result of the combinatorial part provided by the vns
    for f_j, f in zip(range(1,J1),features):
        if FSPp[f_j] == 0:
            m.beta[f].fix(0)
            
            
    
    #
    # Objective function 
    #
    
    def beta_xn_(m, n):
        return sum(m.beta[j] * m.x[n, j] for j in m.j_b0)
    
    def objfunction_(m, GLM):
        if GLM == 'Lin':
            return   (1/N) *sum((m.y[n] - beta_xn_(m, n))**2 for n in m.n)
        
        elif GLM == 'Log':
            # senza approssimazione ma stabile
            return (1 / N) * sum( (1 - m.y[n]) * beta_xn_(m, n) + 
                                 pym.log(1 + pym.exp(-beta_xn_(m, n))) 
                                 for n in m.n )
        elif GLM == 'Poi':
            return sum( pym.exp(beta_xn_(m, n)) - beta_xn_(m, n)*(m.y[n])  for n in m.n )
            


         
    m.objfunction = pym.Objective(rule=objfunction_(m,GLM),  sense=pym.minimize)
    
    Solver = {'Lin': 'gurobi_persistent', 'Log':'ipopt', 'Poi':'ipopt'}.get(GLM)
    
    # 
    # Solver
    #

    solver = pym.SolverFactory(Solver)
    if 'persistent' in Solver:
        solver.set_instance(m)
        solver.solve(tee = True)
    
    else:
        results = solver.solve(m, tee=True)
    

    
    #
    # Print Results
    #
    
    
    print("Coefficients")
    betaP_p = {}
    for f in m.j_b0:
        betaP_p[f] = m.beta[f].value
        print(f"beta[{f}] = {m.beta[f].value}") 
    

    print("\n")
    
    # Recupera il primo (e di solito unico) obiettivo attivo del modello
    for obj in m.component_objects(pym.Objective, active=True):
        print(f"Current obj value ($tau$ ): {pym.value(obj)}")
    
    return pym.value(obj), betaP_p

