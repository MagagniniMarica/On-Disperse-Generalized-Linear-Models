"""
@author: Marica Magagnini
"""

import numpy as np
import random
import copy
from pyomo import environ as pym
import pandas as pd
import itertools

from pyomo.opt import TerminationCondition
from f_print import get_obj_sol_

"""
Note: 
Define a minimum theta of sparsity to avoid checking combinations that
are too few coefficients that would never achieve the required accuracy.
"""
theta_min =5 # Default value, can be adjusted based on the problem context and the dataset.

###############################################################################
# Methods for VNS
###############################################################################
# This function update the VNS index k in [0, .., K-1]
def update_k_(k,K):
    if k < K-1:
        return k+1
    else:
        return 1

# Relative gap computation when available
def rel_gap_(current_obj, optimal_obj, gap_tol ):
    if (optimal_obj is not None) and (gap_tol is not None):
        denom = max(abs(optimal_obj), 1e-10)
        rel_gap = abs(current_obj - optimal_obj) / denom
        print(f"Current relative gap = {rel_gap:.6e}")
        return rel_gap
    else:
        return None

###############################################################################
# General methods for perturbation
###############################################################################

def count_differences(fsp1, fsp2, return_matching=False):
    """
    Compute

        δ(Ξ, Ξ*) = min_{φ in Ω} sum_{p=1}^P || ξ_p - ξ*_{φ(p)} ||_1

    for selecting the minimum distance between two collections of vectors Ξ and Ξ*.

    Parameters
    ----------
    fsp1, fsp2 : iterable di iterable numerici
        Collection of P vectors of dimension d, representing the two feature selection patterns Ξ and Ξ*.
    return_matching : bool, optional
        If true, also return the optimal permutation φ that achieves the minimum distance.

    Returns
    -------
    best_cost : float
        Value of the distance δ(Ξ, Ξ*)
    best_perm : tuple, optional
        Optimal permutation, if return_matching=True
    """

    if fsp1 is None or fsp2 is None:
        raise ValueError("fsp1 and fsp2 cannot be None")

    X = np.asarray(list(fsp1), dtype=float)
    Y = np.asarray(list(fsp2), dtype=float)

    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError("fsp1 and fsp2 must be 2D collections of vectors")

    P1, d1 = X.shape
    P2, d2 = Y.shape

    if P1 != P2:
        raise ValueError(
            f"The two collections must have the same cardinality: {P1} != {P2}"
        )

    if d1 != d2:
        raise ValueError(
            f"The vectors must have the same dimension: {d1} != {d2}"
        )

    P = P1
    best_cost = float("inf")
    best_perm = None

    for perm in itertools.permutations(range(P)):
        cost = sum(np.abs(X[p] - Y[perm[p]]).sum() for p in range(P))

        if cost < best_cost:
            best_cost = cost
            best_perm = perm

    if return_matching:
        return best_cost, best_perm
    return best_cost

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



def generate_valid_list(J1, theta):
    """
    First position is always 1, then  generate a list of J1-1 zeros.
    Add randomly between tau_min and theta ones in the remaining positions.
    """
    lst = [0] * J1
    lst[0] = 1
    sparsity = random.randint(theta_min, theta)
    indices = random.sample(range(1, J1), sparsity)
    for idx in indices:
        lst[idx] = 1
    return tuple(lst)

def random_flip_structure(J1, base_row, n_flips):
    """
    Flippa n_flips bit in posizioni da 1..J1-1.
    First position is always 1, so we don't flip it.
    """
    row = list(base_row)
    n_flips = min(n_flips, J1 - 1)
    flip_positions = random.sample(range(1, J1), n_flips)
    for pos in flip_positions:
        row[pos] = 1 - row[pos]
    row[0] = 1
    return tuple(row)


def canonical_fsp(fsp):
    return tuple(sorted(tuple(row) for row in fsp))

################################
# case specific
################################
def extract_l1_(P,FSP_maxAcc_coeff,SQ, features):
    #INNER DISPERSION 
    # To construct E, I am only interested in the order of the indices, they are in ascending order
    E ={f: np.zeros((P,P)) for f in features}
    
    SP_ = pd.DataFrame(FSP_maxAcc_coeff)
    SP_ord = {}
    for f in features:
        s = SP_[f].sort_values(ascending=True)
        SP_ord[f] = s.reset_index()  # column-wise
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
    # Fixed a feature, Order SQ models in ascending order

    
    SP_SQ_ord = {}
    SP_SQ = pd.concat([SP_, SQ], ignore_index=True)
    
    Q = len(SQ)

    for f in features:
        s = SP_SQ[f].sort_values(ascending=True)
        SP_SQ_ord[f] = s.reset_index()  # columns-wise
        SP_SQ_ord[f].columns = ["PQ", "value"]

   
    # Build T
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
                    #  \beta[p,f] <= \betaSQ[q,f]
                elif a > b and SP_[f].iloc[p] != 0 :
                    T[f][p,q-P] = -1
                    #\beta[p,f] >= \betaSQ[q,f]
    
    return E,T

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
    # array dei prodotti scalari
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


def min_hamming_distance(FSP, SQ):
    
    hamming_distance = 1000
    # inner dsa dispersion
    for r1_idx, r1 in enumerate(FSP):
        for r2 in FSP[r1_idx+1:]:
            hd_r12 = sum(b1 != b2 for b1, b2 in zip(list(r1), list(r2)))
            if hd_r12 < hamming_distance:
                hamming_distance = hd_r12
                
    # outer dsa dispersion 
    eps = 1e-8
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
# Perturbatio 'l1','l2','o1' 
def perturbation_(FSP_store, FSP, theta, k, P,
                  GLM, dataset, target, features, tau, J1,
                  dispersion, SQ, X0, solver_Iterations, trash_valid_fsp):
    
    ###########################################################################
    # Initialization
    ###########################################################################
    coeff_cache = {}
    (V,W,E,T) = (None,None,None,None)
    
    
    
    #
    # Returns the combinatorial stucture according to the dispersion
    # extract by the set of max acc solutions
    #
    def C_(P,FSP_MaxACC_coeff,SQ, features, X0):
        V,W,E,T = (None, None,None,None)

        if  dispersion == 'l1':
            E,T = extract_l1_(P,FSP_MaxACC_coeff,SQ, features)
        elif dispersion == 'o1':
            V,W = extract_o1_(X0, SQ, P,FSP_MaxACC_coeff )
               
        return  V,W,E,T
    
    def get_coeff_for_structure(fsp_row):
        row = tuple(fsp_row)
        
        if row in trash_valid_fsp['trash']:
            return None
    
        if sum(row[1:]) > theta:
            trash_valid_fsp['trash'].add(row)
            return None
        
    
        if row in coeff_cache:
            return coeff_cache[row]
    
        
        if row in trash_valid_fsp['coeff']:
            coeff = trash_valid_fsp['coeff'][row]
            coeff_cache[row] = coeff
            return coeff
    
        fsp_acc, fsp_coeff = acc_requirements( GLM, dataset, target, features, 
                                            list(row), J1, tau )
    
        if fsp_acc:
            trash_valid_fsp['coeff'][row] = fsp_coeff
            coeff_cache[row] = fsp_coeff
            return fsp_coeff
        else:
            trash_valid_fsp['trash'].add(row)
            return None
    
    
    
    def build_initial_solution():
        """
        Build a valid initial solution of P structures.
        """
        selected_rows = []
        selected_coeffs = []
        max_attempts_init = 2500
        attempts = 0


        while attempts < max_attempts_init and len(selected_rows) < P:
            print(f'________Build solution attemps -- {attempts+1} -- __________')
            attempts += 1
            
            valid_pool = list(trash_valid_fsp['coeff'].keys())
            random.shuffle(valid_pool)
            
            use_known = (len(valid_pool) > 0 and random.random() < 0.8)
            
            if use_known:
                cand = random.choice(valid_pool)
            else:
                cand = generate_valid_list(J1, theta)

            coeff = get_coeff_for_structure(cand)
            if coeff is None:
                continue
            
            selected_rows.append(cand)
            selected_coeffs.append(coeff)

        if len(selected_rows) == P:
            fsp_star = tuple(selected_rows)
            fsp_key = canonical_fsp(fsp_star)
        
            if fsp_key not in FSP_store:
                FSP_store.add(fsp_key)
                return fsp_star, selected_coeffs

        return None, None
    
    
    
   
    ###########################################################################
    # CASE 1: no incumbent
    ###########################################################################
    if not FSP:
        fsp_star, coeffs = build_initial_solution()
        if fsp_star is not None:
            (V,W,E,T) = C_(P,coeffs,SQ, features, X0)
            return fsp_star, coeffs,V,W,E,T, trash_valid_fsp, FSP_store
        return None, None,V,W,E,T, trash_valid_fsp, FSP_store
    
    ###########################################################################
    # CASE 2: perturb incumbent with global bit-distance controlled by k
    ###########################################################################

    current_fsp = tuple(tuple(row) for row in FSP)
    
    
    max_attempts = 150
    for attempt in range(max_attempts):
        print(f"\n-------------\n attempt : {attempt+1}\n-------------\n")

        candidate_fsp = list(current_fsp)

        # budget of flip "proposed"; then we will control the true distance with count_differences
        proposed_flips = 0
        local_attempts = 0
        max_local_attempts = 50

        while proposed_flips < k and local_attempts < max_local_attempts:
            local_attempts += 1

            # Pick a row to perturb
            p = random.randrange(P)
            base_row = candidate_fsp[p]

            # number of elementary flips to attempt on this row
            flips_here = random.randint(1, max(1, min(k - proposed_flips, J1 - 1)))

            # first, try a local flip on the current row
            cand_row = random_flip_structure(J1, base_row, flips_here)
            coeff = get_coeff_for_structure(cand_row)

            #Fallback: if it doesn't work, try to pick a valid structure that is already known
            if coeff is None and len(trash_valid_fsp['coeff']) > 0:
                valid_pool = list(trash_valid_fsp['coeff'].keys())
                random.shuffle(valid_pool)
                
                replaced = False
                for row_known in valid_pool:
                    if  row_known in [row for i, row in enumerate(candidate_fsp) if i != p]:
                        continue
                    
                    candidate_fsp[p] = row_known
                    replaced = True
                    break
                
                if replaced:
                    proposed_flips += flips_here
                    continue
                else:
                    continue

            if coeff is None:
                continue


            candidate_fsp[p] = cand_row
            proposed_flips += flips_here

        candidate_fsp = tuple(candidate_fsp)
        candidate_key = canonical_fsp(candidate_fsp)

        if candidate_key in FSP_store:
            continue

        true_dist = count_differences(current_fsp, candidate_fsp)

        if true_dist < k:
            continue
    

        coeffs = []
        feasible = True
        for row in candidate_fsp:
            coeff = get_coeff_for_structure(row)
            if coeff is None:
                feasible = False
                break
            coeffs.append(coeff)

        if not feasible:
            continue
        
        FSP_store.add(candidate_key)
        (V,W,E,T) = C_(P,coeffs,SQ, features, X0)
        return candidate_fsp, coeffs, V,W,E,T, trash_valid_fsp, FSP_store

    ###########################################################################
    # No candidate found
    ###########################################################################
    return None, None,V,W,E,T, trash_valid_fsp, FSP_store
    
    
    

def perturbation_dsa_(GLM, dataset, target, features, P, J1, k, tau, theta, gamma,
                      SQ, FSP, FSP_store, trash_valid_fsp):

    ###########################################################################
    # Initialization
    ###########################################################################

    eps = 1e-8
    FSQ = [tuple(0 if abs(x) < eps else 1 for x in SQ.iloc[q]) for q in range(len(SQ))]
    coeff_cache = {}

    ###########################################################################
    # Helpers
    ###########################################################################


    def hamming_distance(list1, list2):
        return sum(b1 != b2 for b1, b2 in zip(list1, list2))
    
    # Filter rows of the current solution at a distance greater than the current one from SQ
    def row_ok_wrt_SQ(row, fsq, gamma):
        return min(hamming_distance(row, fsq_row) for fsq_row in fsq) > gamma

    

    def satisfies_gamma_constraint(fsp_star, fsq, gamma):
        # Outer dispersion wrt SQ
        for star_row in fsp_star:
            for fsq_row in fsq:
                if hamming_distance(star_row, fsq_row) <= gamma:
                    return False

        # Inner dispersion inside FSP_star
        for i in range(len(fsp_star)):
            for j in range(i + 1, len(fsp_star)):
                if hamming_distance(fsp_star[i], fsp_star[j]) <= gamma:
                    return False

        return True

    
    def get_coeff_for_structure(fsp_row):
        row = tuple(fsp_row)
        
        if row in trash_valid_fsp['trash']:
            return None
    
        if sum(row[1:]) > theta:
            trash_valid_fsp['trash'].add(row)
            return None
        
        if not row_ok_wrt_SQ(row, FSQ, gamma):
            trash_valid_fsp['trash'].add(row)
            return None
        
    
        if row in coeff_cache:
            return coeff_cache[row]
    
        
        if row in trash_valid_fsp['coeff']:
            coeff = trash_valid_fsp['coeff'][row]
            coeff_cache[row] = coeff
            return coeff
    
        fsp_acc, fsp_coeff = acc_requirements( GLM, dataset, target, features, 
                                            list(row), J1, tau )
    
        if fsp_acc:
            trash_valid_fsp['coeff'][row] = fsp_coeff
            coeff_cache[row] = fsp_coeff
            return fsp_coeff
        else:
            trash_valid_fsp['trash'].add(row)
            return None

    def build_initial_solution():
        """
        Builds an initial solution of P valid structures.
        """
        selected_rows = []
        selected_coeffs = []
        max_attempts_init = 2500
        attempts = 0
        failed_extensions = 0

        while attempts < max_attempts_init and len(selected_rows) < P:
            print(f'________Build solution attemps -- {attempts+1} -- __________')
            attempts += 1

            
            
            valid_known_rows = [
                row for row in trash_valid_fsp['coeff'].keys()
                if min(hamming_distance(row, fsq_row) for fsq_row in FSQ) > gamma
                    ]
            use_known = (len(valid_known_rows) > 0 and random.random() < 0.8)
            
            if use_known:
                cand = random.choice(valid_known_rows)
            else:
                cand = generate_valid_list(J1, theta)

            coeff = get_coeff_for_structure(cand)
            if coeff is None:
                continue

            if cand in selected_rows:
                continue

            trial = selected_rows + [cand]
            if satisfies_gamma_constraint(trial, FSQ, gamma):
                selected_rows.append(cand)
                selected_coeffs.append(coeff)
                failed_extensions = 0
            else:
                failed_extensions += 1
            
                if failed_extensions >= 5:
                    selected_rows = []
                    selected_coeffs = []
                    failed_extensions = 0
                
            if len(selected_rows) == P:
                break

        if len(selected_rows) == P:
            fsp_star = tuple(selected_rows)
            fsp_key = canonical_fsp(fsp_star)
        
            if fsp_key not in FSP_store:
                FSP_store.add(fsp_key)
                return fsp_star, selected_coeffs

        return None, None

    

    def complete_partial_solution(partial_rows):
        selected_rows = list(partial_rows)
        selected_coeffs = []
    
        for row in selected_rows:
            coeff = get_coeff_for_structure(row)
            if coeff is None:
                return None, None
            selected_coeffs.append(coeff)
    
        valid_pool = [
            row for row in trash_valid_fsp['coeff'].keys()
            if row not in selected_rows and row_ok_wrt_SQ(row, FSQ, gamma)
        ]
    
        valid_pool.sort(
            key=lambda row: min(hamming_distance(row, fsq_row) for fsq_row in FSQ),
            reverse=True
        )
    
        for cand in valid_pool:
            if len(selected_rows) >= P:
                break
    
            trial = selected_rows + [cand]
            if satisfies_gamma_constraint(trial, FSQ, gamma):
                selected_rows.append(cand)
                selected_coeffs.append(trash_valid_fsp['coeff'][cand])
    
        if len(selected_rows) == P:
            fsp_completed = tuple(selected_rows)
            fsp_key = canonical_fsp(fsp_completed)
    
            if fsp_key not in FSP_store:
                FSP_store.add(fsp_key)
                return fsp_completed, selected_coeffs
    
        return None, None
    ###########################################################################
    # CASE 1: no incumbent
    ###########################################################################
    if not FSP:
        fsp_star, coeffs = build_initial_solution()
        if fsp_star is not None:
            return fsp_star, coeffs, trash_valid_fsp, FSP_store
        return None, None, trash_valid_fsp, FSP_store

    ###########################################################################
    # CASE 2: perturb incumbent with global bit-distance controlled by k
    ###########################################################################

    current_fsp = tuple(tuple(row) for row in FSP)

    
    
    current_rows_ok_wrt_SQ = tuple(
                                    row for row in current_fsp
                                    if row_ok_wrt_SQ(row, FSQ, gamma)
                                        )
    
    if len(current_rows_ok_wrt_SQ) == 0:
        fsp_star, coeffs = build_initial_solution()
        if fsp_star is not None:
            return fsp_star, coeffs, trash_valid_fsp, FSP_store
        return None, None, trash_valid_fsp, FSP_store
    
    if len(current_rows_ok_wrt_SQ) < P:
        completed_fsp, completed_coeffs = complete_partial_solution(current_rows_ok_wrt_SQ)
        if completed_fsp is not None:
            current_fsp = completed_fsp
        else:
            fsp_star, coeffs = build_initial_solution()
            if fsp_star is not None:
                return fsp_star, coeffs, trash_valid_fsp, FSP_store
            return None, None, trash_valid_fsp, FSP_store
    else:
        current_fsp = tuple(current_rows_ok_wrt_SQ)

    max_attempts = 150
    for attempt in range(max_attempts):
        print(f"\n-------------\n attempt : {attempt+1}\n-------------\n")

        candidate_fsp = list(current_fsp)

        # budget of flip "proposed"; then we will control the true distance with count_differences
        proposed_flips = 0
        local_attempts = 0
        max_local_attempts = 50

        while proposed_flips < k and local_attempts < max_local_attempts:
            local_attempts += 1

            # Pick a row to perturb
            p = random.randrange(P)
            base_row = candidate_fsp[p]

            # number of elementary flips to attempt on this row
            flips_here = random.randint(1, max(1, min(k - proposed_flips, J1 - 1)))

            # first, try a local flip on the current row
            cand_row = random_flip_structure(J1, base_row, flips_here)
            coeff = get_coeff_for_structure(cand_row)

            # if it doesn't work, try to pick a valid structure that is already known
            if coeff is None and len(trash_valid_fsp['coeff']) > 0:
             
                valid_pool = [
                                row_known for row_known in trash_valid_fsp['coeff'].keys()
                                if row_known != base_row and row_ok_wrt_SQ(row_known, FSQ, gamma)
                            ]
            
                valid_pool.sort(
                    key=lambda row_known: (
                        hamming_distance(row_known, base_row),
                        min(hamming_distance(row_known, fsq_row) for fsq_row in FSQ)
                    ),
                    reverse=True
                )
                
                
                replaced = False
                for row_known in valid_pool:
                    if row_known == base_row:
                        continue

                    trial_fsp = candidate_fsp.copy()
                    trial_fsp[p] = row_known

                    if satisfies_gamma_constraint(trial_fsp, FSQ, gamma):
                        candidate_fsp[p] = row_known
                        replaced = True
                        break

                if replaced:
                    proposed_flips += flips_here
                    continue
                else:
                    continue

            if coeff is None:
                continue

            trial_fsp = candidate_fsp.copy()
            trial_fsp[p] = cand_row

            if not satisfies_gamma_constraint(trial_fsp, FSQ, gamma):
                continue

            candidate_fsp[p] = cand_row
            proposed_flips += flips_here

        candidate_fsp = tuple(candidate_fsp)
        candidate_key = canonical_fsp(candidate_fsp)

        if candidate_key in FSP_store:
            continue

        if not satisfies_gamma_constraint(candidate_fsp, FSQ, gamma):
            continue

        true_dist = count_differences(current_fsp, candidate_fsp)

        if true_dist < k:
            continue
    

        coeffs = []
        feasible = True
        for row in candidate_fsp:
            coeff = get_coeff_for_structure(row)
            if coeff is None:
                feasible = False
                break
            coeffs.append(coeff)

        if not feasible:
            continue

        FSP_store.add(candidate_key)
        return candidate_fsp, coeffs, trash_valid_fsp, FSP_store

    ###########################################################################
    # No candidate found
    ###########################################################################
    return None, None, trash_valid_fsp, FSP_store


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
    dataset_bias['bias'] = 1.0
    x_dict = {(n, j): dataset_bias.iloc[n, dataset_bias.columns.get_loc(j)]
          for n in range(len(dataset_bias))
          for j in dataset_bias.columns}
    m.x = pym.Param(m.n, m.j_b0, initialize=x_dict, mutable=True, within = pym.Reals)
    
    
    #
    # variables
    #
    ulb = (-30,30) if len(features) <20 else (-50,50) if  (len(features) > 20 and len(features) < 70) else (-1e5, 1e5)
    m.beta = pym.Var(m.j_b0, within=pym.Reals, bounds=ulb) #(-30,30) BH, (-50,50) ITT, (-1e5, 1e5) CC
    
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
    
   
    for obj in m.component_objects(pym.Objective, active=True):
        print(f"Current obj value ($tau$ ): {pym.value(obj)}")
    
    return pym.value(obj), betaP_p

