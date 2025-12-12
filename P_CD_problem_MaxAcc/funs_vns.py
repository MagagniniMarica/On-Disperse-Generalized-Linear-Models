# -*- coding: utf-8 -*-
"""

@author: Marica Magagnini

This file contains the support fuction for the heuristic - VNS strategy - in 
VNS_Max_Acc.py

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
import time as tm
from P_CD_problem_MaxAcc.PCD_MAX_dispersion_VNS_ import P_max_disp_
from pyomo.opt import TerminationCondition
from f_print import get_obj_sol_

# This function update the VNS
def update_k_(k,K):
    if k < K-1:
        return k+1
    else:
        return 1


# 'Classical Hamming distance' between two feature selection structures (\xi^p1,\xi^p2)
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
# as feasible solution of the maximal dispersion problem.
#
def acc_requirements(GLM, dataset, target, features, FSP_p, J1, tau):
    FSP_p_loss, FSP_p_coeff = max_acc_p_(GLM, dataset, target, features, FSP_p, J1)
    if FSP_p_loss <= tau:
        return True, FSP_p_coeff
    else:
        print('Combinatorial attemp not satisfying accuracy requirements.')
        return False, []




# This function merge two list mantaining thier inner order
def shuffle_merge_ordered(A, B):
    result = []
    i, j = 0, 0
    total = len(A) + len(B)
    while len(result) < total:
        # scegli casualmente se prendere da A o B, se possibile
        if i < len(A) and j < len(B):
            if random.random() < 0.5:
                result.append(A[i])
                i += 1
            else:
                result.append(B[j])
                j += 1
        elif i < len(A):
            result.append(A[i])
            i += 1
        else:
            result.append(B[j])
            j += 1
    return result

# This function provides the combinatorial part in case of l1 norm between outputs (o1). 
# It selects a randorm ordered list of P U Q models, storing the order infomation in 
# matrices 'v' and 'w' . See Algorithm 4 in Supplementary Material.
def o1_dispersion_(x0, B0, P ):
    
    # INNER DISPERSION v[i,j] = 1 if beta_i >= beta_j + gamma i,j = 1..P i!=j 
    SP_ord = np.random.permutation(np.arange(0, P)).tolist()
    v = np.zeros((P,P))
    for r in range(P):
        for c in range(P):
            if SP_ord.index(r) < SP_ord.index(c):
                v[r,c]=1
    
    # OUTER DISPERSION w[i,j] = 1 if beta_i >= beta_j + gamma i = 1,..P, j=1, .., Q
    Q = len(B0)
    

    if Q>1:
        SQ_x0 = np.array([np.dot(beta, x0) for beta in B0])
    else:
        SQ_x0 = np.dot(B0,x0)
    
    
    # Descending order 
    ordered_indices = (-SQ_x0).argsort()
    SQ_ord = [x+P for x in ordered_indices]  # to list
    
    SP_SQ = shuffle_merge_ordered(SQ_ord, SP_ord)
    
    w = np.zeros((P,Q))
    for r in range(P):
        for c in range(P,P+Q):
            if SP_SQ.index(r) < SP_SQ.index(c):
                w[r, c-P] = 1
    
    return v,w


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
   

# Given a set of P models which satisfy the accuracy constraints, this function reconstructs 
# its combinatorial structure with respect to 'l1' coeffients dispersion. See Algorithm 4, (d1, alpha =1)
# in the Supplementary Material
def extract_l1_(P,FSP_maxAcc_coeff,SQ, features):
    #INNER DISPERSION 
    # per la costruzione di E mi intressano solo gli ordini degli indici,
    # sono in ordine crescente 
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
    
    # OUTER DISPERSION
    # fissata una feature, ordino i modelli SQ in ordine crescente
    
    SP_SQ_ord = {}
    SP_SQ = pd.concat([SP_, SQ], ignore_index=True)
    
    Q = len(SQ)

    for f in features:
        s = SP_SQ[f].sort_values(ascending=True)
        SP_SQ_ord[f] = s.reset_index()  # l’indice originale va in una colonna
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
                    # cioè \beta[p,f] <= \B0[q,f]
                elif a > b and SP_[f].iloc[p] != 0 :
                    T[f][p,q-P] = -1
                    #\beta[p,f] >= \B0[q,f]
    
    return E,T


# This function provides the combinatorial part in case of l1 norm between coefficients. 
# For each feature j, it selects a randorm ordered list of P U Q models, 
#storing the order infomation in 
# dicts 'E' and 'T' . See Algorithm 4 in Supplementary Material.
def l1_dispersion_(P,FSP,B0, features):
    # Note: pay attention to the FSP structure
    
    FSP_DF =  pd.DataFrame(FSP, columns = pd.Index(['bias']).append(features))
    
    # INNER DISPERSION  

    SP_ord = {}
    for f in features:
        s = FSP_DF[f]
        zeros = s[s == 0]  # Series of 0s
        ones = s[s == 1]   # Series of 1s

  
        ones_shuffled = ones.sample(frac=1)  # original indices
        n_ones_before = np.random.randint(0, len(ones_shuffled)+1)
        ones_before = ones_shuffled.iloc[:n_ones_before]
        ones_after = ones_shuffled.iloc[n_ones_before:]
        ones_before = ones_before.replace(1, -1)
        
        s = pd.concat([ones_before, zeros, ones_after])
        SP_ord[f] = s.reset_index()  
        SP_ord[f].columns = ["P", "value"]



    # ! indices are in ascending order 
    E ={f: np.zeros((P,P)) for f in features}
    
  
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
                
    """
    Note:
    If E[f][p,p'] = 1
            then \beta[p,f] <= \beta[p',f]
    If E[f][p,p'] = -1
            then \beta[p,f] >= \beta[p',f]
    If E[f][p,p'] = 0
            then \beta[p,f] == \beta[p',f] == 0 per p !=p' o p == p'
    """
    
    
    # OUTER DISPERSION
    # Sort B_0 models in ascending order
    Q = len(B0)
    SQ_ord = {}
    SQ = B0.copy()
    SQ.index = [P+q for q in SQ.index]

    for f in features:
        s = SQ[f].sort_values(ascending=True)
        SQ_ord[f] = s.reset_index() 
        SQ_ord[f].columns = ["Q", "value"]

   
    
    # Merge SP_ord and  SQ_ord  mantaining their inner orders previously computed 
    # and respercting B_0 values
    merged = {}

    for f in features:
        sp_f = SP_ord[f]
        sq_f = SQ_ord[f]

        # Split
        sp_zeros = sp_f[sp_f['value'] == 0]
        sp_pos = sp_f[sp_f['value'] > 0]
        sp_neg = sp_f[sp_f['value'] < 0]

        # positives and negatives
        sq_pos = sq_f[sq_f['value'] > 0]
        sq_neg = sq_f[sq_f['value'] < 0]

        # shuffle merge each
        pos_merged = shuffle_merge_ordered(sp_pos['P'].to_list(),sq_pos['Q'].to_list())
        neg_merged = shuffle_merge_ordered(sp_neg['P'].to_list(),sq_neg['Q'].to_list())

        #merge
        blocks = neg_merged+ sp_zeros['P'].to_list() + pos_merged

        merged[f] = blocks

    
    # Build T
    T ={f: np.zeros((P,Q)) for f in features}
   
    for f in features:
        for p in range(P):
            for q in range(P, P+Q):
                if merged[f].index(p) < merged[f].index(q) and FSP_DF[f].iloc[p] != 0 :
                    T[f][p,q-P] = 1
                    # cioè \beta[p,f] <= \B0[q,]
                elif merged[f].index(p) > merged[f].index(q) and FSP_DF[f].iloc[p] != 0 :
                    T[f][p,q-P] = -1
                    #\beta[p,f] >= \B0[q,]
                
    
    
    return E,T    

###############################################################################
# PERTURBATIONS
###############################################################################    
# PERTURBATION 'l2','l1','o1'
def perturbation_(FSP_store, FSP, theta, k, P,
                  GLM, dataset, target, features, tau, J1,
                  dispersion, SQ, X0, gamma, solver_Iterations, trash_valid_fsp):
    def generate_valid_list(J1, theta, ref_list, n_flip):
        
        max_n_flip = 1
        if ref_list != [] :
            
            lst = ref_list.copy()
            indices = list(range(1, len(lst)))

            # Split 0s and 1s indices
            zeros = [i for i in indices if lst[i] == 0]
            ones = [i for i in indices if lst[i] == 1]
            
            #  Maximum feasible exchange
            max_n_flip = min(len(zeros), len(ones))
            
        if n_flip < max_n_flip : 
                # Random choice
                chosen_zeros = random.sample(zeros, n_flip)
                chosen_ones = random.sample(ones, n_flip)

                # Exchange
                for i0, i1 in zip(chosen_zeros, chosen_ones):
                    lst[i0], lst[i1] = lst[i1], lst[i0]
        
        if ref_list == [] or  n_flip >= max_n_flip: # costruzione casuale
            lst = [0] * J1
            lst[0] = 1  # il primo elemento è sempre 1
            sparsity = random.randint(5,theta) ## In general random.randint(1, theta)
            indices = random.sample(range(1, J1), sparsity)
            for idx in indices:
                lst[idx] = 1

        return lst

    #
    # Control that the feature selected can reach the dipsersion requirements.
    # If not, build another one. If yes, hold the solution cofficients 
    # as feasible solution of the maximal accuracy problem
    def disp_requirements(GLM, dispersion, dataset, target, features, FSP,SQ, X0, J1, gamma,tau, solver_Iterations):
        comb_iteretions = 1 if dispersion == 'l2' or dispersion == 'o1' else 5
        V,W,E,T = (None, None,None,None)
        V_,W_,E_,T_ = (None, None,None,None)
        obj_star, FSP_MaxDisp_coeff = (0,[])
        D_requirement = False
        term_cond = None
        
        #
        # coefficienti di massima accuratezza per FSP
        #
        FSP_MaxACC_coeff = []
        for fsp_p in FSP:
            acc_p, FSP_p_MaxACC_coeff= acc_requirements(GLM, dataset, target, features, fsp_p, J1, tau)
            if acc_p:
                FSP_MaxACC_coeff.append(FSP_p_MaxACC_coeff)
            else:
                print('------------ERRORE (disp_requirments line 255)-------------')
                return D_requirement, FSP_MaxDisp_coeff, V,W,E,T
        #
        #
        #
        
        for _ in range(comb_iteretions):
            # combinatorial part for l1 and o1
            if  dispersion == 'l1':
                if (E_,T_) == (None,None):
                    E_,T_ = extract_l1_(P,FSP_MaxACC_coeff,SQ, features)
                else:
                    E_,T_ = l1_dispersion_(P,FSP,SQ, features)

            elif dispersion == 'o1':
                if V_ is None and W_ is None:
                    V_,W_ =  extract_o1_(X0, SQ, P,FSP_MaxACC_coeff )
                else:
                    V_,W_ = o1_dispersion_(X0, SQ, P )

            else:
                V_,W_,E_,T_ = (None, None,None,None) 
            #
            # Local D-PDP-GLM ---> Mxx dipersion problem 
            #
            instance = P_max_disp_(GLM,dispersion, dataset, target, features, FSP,
                                      SQ, P, tau, X0, 
                                      V = V_,W = W_, T=T_, E=E_)
            
            #
            # === Provide Initial Guess for Ipopt === the one with maximal accuracy
            #
            for p in range(P):
                for f in instance.j_b0:     
                    instance.beta[p, f].value = FSP_MaxACC_coeff[p][f]  
                
            
            #
            # === Solve with Ipopt ===
            #
            solver = pym.SolverFactory('ipopt')
            
            solver.options['print_level'] = 5               # Controls verbosity (0 = silent, 5 = detailed)
            solver.options['max_iter'] = solver_Iterations  # Max iterations
            solver.options['tol'] = 1e-6                    # Convergence tolerance
            solver.options['acceptable_tol'] = 1e-5
            
            ts = tm.time()
            try:
                
                results = solver.solve(instance, tee=True)
                term_cond  = results.solver.termination_condition
            
                t = tm.time() -ts
            
                # Check dispersion requirements and the new solution is better than the previous one
                if term_cond == TerminationCondition.optimal or term_cond == TerminationCondition.feasible:
                
                    obj_, betaP_ = get_obj_sol_(instance)
                    
                    if obj_ >= gamma and obj_ > obj_star:
                        obj_star, FSP_MaxDisp_coeff = (obj_, betaP_)
                        V,W,E,T  = (V_,W_,E_,T_)
                        D_requirement =  True
                        print("[OPT] New Max Dispersion solution.")
                
                else:
                    print("[WARN] No solution found.")
                    
            except Exception as e:
                print(f"[ERROR] Error Ipopt solver: {e}")
                t = tm.time() -ts
 
            print(f'Time computation : {t}')
        
        return D_requirement, FSP_MaxDisp_coeff, V,W,E,T
    
    
 
    (V,W,E,T) = (None,None,None,None)
    
    #
    # Comination of feasible availabe structures not already considered
    #
    combo_unique_tuples = list(itertools.combinations_with_replacement(trash_valid_fsp['valid'], P))
    for FSP_star in combo_unique_tuples:
        # VNS perturbation -> FSP_star has k elements different from FSP
        if (count_differences(FSP, FSP_star) >= k or FSP== [] ) and FSP_star not in FSP_store['valid'] and FSP_star not in FSP_store['trash']:
            
            D_req, FSP_maxDisp_Coeff, V,W,E,T = disp_requirements(GLM, dispersion,
                                                                  dataset, target, features, 
                                                                  FSP_star,SQ, X0, J1, gamma, tau,
                                                                  solver_Iterations)
            if D_req:
                FSP_store['valid'].add(FSP_star)
                return FSP_star, FSP_maxDisp_Coeff, V,W,E,T, trash_valid_fsp,FSP_store
            
            else:
                FSP_store['trash'].add(FSP_star)

        
    #
    # New construction
    #
    ref_list = []
    # Choose a reference structure
    if trash_valid_fsp['valid'] :
        ref_list = list(list(trash_valid_fsp['valid'])[-1])

    
        
    n_dst = 0       # number of distinct structure built

  
    # Generation phase
    FSPp = generate_valid_list(J1, theta, ref_list, k)  
    
    # Check if FSPp is already known
    if tuple(FSPp) not in trash_valid_fsp['trash'] and tuple(FSPp) not in trash_valid_fsp['valid'] :
        acc_p, FSPp_MaxACC_coeff= acc_requirements(GLM, dataset, target, features, FSPp, J1, tau)
    else: 

        acc_p =False
          
        
    #   Check accuracy constraints: if satisfied ---> allocate to valid structures
    if  acc_p:                   
        n_dst+=1
        trash_valid_fsp['valid'].add(tuple(FSPp))
        print('Found!')
        
            
        # Generate all the combinations using the incremented set of valid structures
        combo_dst = list(itertools.combinations_with_replacement(trash_valid_fsp['valid'], P))
        for FSP_star in combo_dst:

            if FSP_star not in FSP_store['valid'] and FSP_star not in FSP_store['trash'] and (FSP == [] or count_differences(FSP, FSP_star) >= k):
                # Check dispersion constraints
                D_requirement, FSP_MaxDisp_coeff, V,W,E,T = disp_requirements(GLM, dispersion,
                                                                              dataset, target,
                                                                              features, FSP_star,
                                                                              SQ, X0, J1, gamma,tau,
                                                                              solver_Iterations)
                # If dispersion constraints are satisfied
                if D_requirement:
                    print('Dispersion requirement satisfied.')
                    FSP_store['valid'].add(FSP_star)
                    return FSP_star, FSP_MaxDisp_coeff,V,W,E,T,  trash_valid_fsp, FSP_store
                else:
                    FSP_store['trash'].add(FSP_star)


        
    #If FSPp is known or it does not satisfy the accuracy constraints ---> trash
    else: 
        trash_valid_fsp['trash'].add(tuple(FSPp))

                    

    return None,None,V,W,E,T, trash_valid_fsp, FSP_store
   
        
        

# Perturbation for dsa dispersion
def perturbation_dsa_(GLM, dataset, target, features,P,J1,k,tau, theta, gamma,SQ,FSP,FSP_store,trash_valid_fsp ):
    
    def generate_valid_list(J1, theta):  
        # Random generation
        lst = [0] * J1
        lst[0] = 1 
        sparsity = random.randint(5,theta) 
        indices = random.sample(range(1, J1), sparsity)
        for idx in indices:
            lst[idx] = 1
        return lst
    
    

    # for dsa dispersion (classical Hamming distance) 
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
    
    eps = 1e-9
    FSQ = [[0 if abs(x) < eps else 1 for x in SQ.iloc[q]] for q in range(len(SQ))]
    
    
    #
    # Comination of feasible availabe structures not already considered
    #
    combo_unique_tuples = list(itertools.combinations_with_replacement(trash_valid_fsp['valid'], P))
    for FSP_star in combo_unique_tuples:
        # VNS perturbation -> FSP_star has k elements different from FSP
        if (FSP == [] or count_differences(FSP, FSP_star) >= k) and FSP_star not in FSP_store['valid'] and FSP_star not in FSP_store['trash'] and satisfies_gamma_constraint(FSP_star, FSQ, gamma) :
            FSP_store['valid'].add(FSP_star)
            for p in range(P):
                FSPp_acc, FSPp_coeff = max_acc_p_(GLM, dataset, target, features, FSP_star[p], J1)
                FSP_maxAcc_Coeff.append(FSPp_coeff)
        
            return FSP_star, FSP_maxAcc_Coeff, trash_valid_fsp, FSP_store
    
        else:
            FSP_store['trash'].add(FSP_star)



    # New costruction
    n_dst = 0       #Number of new distinct structures
    dst = list(trash_valid_fsp['valid'])        # stuctures 
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
                
            # sort
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
                # VNS perturbation -> FSP_star has k elements different from FSP
                if FSP_star not in FSP_store['valid'] and FSP_star not in FSP_store['trash'] and (FSP == [] or count_differences(FSP, FSP_star) >= k) and (satisfies_gamma_constraint(FSP_star, FSQ, gamma)):
                    FSP_store['valid'].add(FSP_star)
                    return FSP_star, FSP_maxAcc_Coeff, trash_valid_fsp, FSP_store
                else:
                    FSP_store['trash'].add(FSP_star)
       
            
    
    return None, None, trash_valid_fsp, FSP_store

def abs_betaj_epsilon(B, epsilon):
    """
    This function check if all non zero features have absolute value greater than epsilon
    """
    for p, beta_dict in B.items():
        for f, value in beta_dict.items():
            # Controllo del valore assoluto
            if abs(value) < epsilon:
                return False
    return True
###############################################################################
#Accuracy check function
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
    # Collega il modello all'istanza del solver
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
