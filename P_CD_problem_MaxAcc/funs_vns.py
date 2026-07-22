"""
@author: Marica Magagnini
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
    Calcola

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

def extract_l1_(P,FSP_maxAcc_coeff,SQ, features):
    #INNER DISPERSION 
    # To construct E, I am only interested in the order of the indices, they are in ascending order
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
    # Fixed a feature, Order SQ models in ascending order
    
    SP_SQ_ord = {}
    SP_SQ = pd.concat([SP_, SQ], ignore_index=True)
    
    Q = len(SQ)

    for f in features:
        s = SP_SQ[f].sort_values(ascending=True)
        SP_SQ_ord[f] = s.reset_index()  # l’indice originale va in una colonna
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


###############################################################################
# PERTURBATIONS
###############################################################################    
# PERTURBATION 'l2','l1','o1'
def perturbation_(FSP_store, FSP, theta, k, P,
                  GLM, dataset, target, features, tau, J1,
                  dispersion, SQ, X0, gamma, solver_Iterations, trash_valid_fsp):


    coeff_cache = {}
    (V,W,E,T) = (None,None,None,None)

    # trash_valid_fsp["trash"]  → not accurate structures
    # trash_valid_fsp["coeff"]  → accurate structures with their coefficients
    def get_coeff_for_structure(fsp_row):
        row = tuple(fsp_row)

        if row in trash_valid_fsp["trash"]:
            return None

        if sum(row[1:]) > theta:
            trash_valid_fsp["trash"].add(row)
            return None

        if row in coeff_cache:
            return coeff_cache[row]

        if row in trash_valid_fsp["coeff"]:
            coeff = trash_valid_fsp["coeff"][row]
            coeff_cache[row] = coeff
            return coeff

        fsp_acc, coeff = acc_requirements(
            GLM,
            dataset,
            target,
            features,
            list(row),
            J1,
            tau,
        )

        if not fsp_acc:
            trash_valid_fsp["trash"].add(row)
            return None

        trash_valid_fsp["coeff"][row] = coeff
        coeff_cache[row] = coeff
        return coeff
    
    #
    # Control that the feature selected can reach the dipsersion requirements.
    # If not, build another one. If yes, hold the solution cofficients 
    # as feasible solution of the maximal dispersion problem
    def disp_requirements(FSP):


        FSP_MaxACC_coeff = []

        for row in FSP:
            coeff = get_coeff_for_structure(row)

            if coeff is None:
                return False, [], None, None, None, None

            FSP_MaxACC_coeff.append(coeff)

        V, W, E, T = (None, None, None, None)
        
        if dispersion == "l1":
            E, T = extract_l1_(P, FSP_MaxACC_coeff, SQ, features)

        elif dispersion == "o1":
            V, W = extract_o1_(X0, SQ, P, FSP_MaxACC_coeff)

        #
        # Risolvo il problema di massima dispersione e vedo se è feasible 
        #
        instance = P_max_disp_(GLM,dispersion, dataset, target, features, FSP,
                                    SQ, P, tau, X0, 
                                    V = V,W = W, T=T, E=E)
        
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
        except Exception as error:
            print(f"[ERROR] Error during the resolution with Ipopt: {error}")
            return False, [], None, None, None, None

        if results.solver.termination_condition not in [
            TerminationCondition.optimal,
            TerminationCondition.feasible,
            ]:
            return False, [], None, None, None, None


        dispersion_value, FSP_MaxDisp_coeff = get_obj_sol_(instance)
        if dispersion_value < gamma:
            return False, [], None, None, None, None

        print(f'Time computation : {tm.time() - ts:.2f} seconds')
    
        return True, FSP_MaxDisp_coeff, V,W,E,T
    
    
    def evaluate_candidate(candidate_fsp):
        candidate_fsp = tuple(tuple(row) for row in candidate_fsp)
        candidate_key = canonical_fsp(candidate_fsp)

        #Known candidate
        if candidate_key in FSP_store["trash"]:
            return None

        if candidate_key in FSP_store["valid"]:
            return None

        #New candidate, check accuracy and dispersion requirements
        valid, coefficients, V, W, E, T = disp_requirements(candidate_fsp)

        if not valid:
            FSP_store["trash"].add(candidate_key)
            return None

        candidate_data = {
            "fsp": candidate_fsp,
            "coeff": coefficients,
            "V": V,
            "W": W,
            "E": E,
            "T": T,
        }

        FSP_store["valid"][candidate_key] = candidate_data
        return candidate_data


    def unpack_candidate(candidate_data):
        return (
            candidate_data["fsp"],
            candidate_data["coeff"],
            candidate_data["V"],
            candidate_data["W"],
            candidate_data["E"],
            candidate_data["T"],
            trash_valid_fsp,
            FSP_store,
        )


    def build_initial_solution():
        """
        Build an initial set of P accurate structures that also satisfies
        the dispersion requirement.
        """
        selected_rows = []
        attempts = 0
        max_attempts_init = 2500

        while attempts < max_attempts_init:
            attempts += 1
            print(
                f"________Build solution attempt "
                f"-- {attempts} -- __________"
            )

            valid_pool = list(trash_valid_fsp["coeff"].keys())
            use_known = bool(valid_pool) and random.random() < 0.8

            if use_known:
                cand = random.choice(valid_pool)
            else:
                cand = generate_valid_list(J1, theta)

            coeff = get_coeff_for_structure(cand)

            if coeff is None:
                continue

            selected_rows.append(tuple(cand))

            if len(selected_rows) < P:
                continue

            candidate_data = evaluate_candidate(tuple(selected_rows))

            if candidate_data is not None:
                return candidate_data

            # The rows are individually accurate, but their combination
            # does not satisfy the dispersion requirement.
            selected_rows = []

        return None


    ###########################################################################
    # CASE 1: no incumbent
    ###########################################################################
    if not FSP:
        candidate_data = build_initial_solution()

        if candidate_data is not None:
            return unpack_candidate(candidate_data)

        return (
            None,
            None,
            V,
            W,
            E,
            T,
            trash_valid_fsp,
            FSP_store,
        )


    ###########################################################################
    # CASE 2: perturb incumbent with global bit-distance controlled by k
    ###########################################################################
    current_fsp = tuple(tuple(row) for row in FSP)

    max_attempts = 150

    for attempt in range(max_attempts):
        print(
            f"\n-------------\n"
            f" attempt : {attempt + 1}\n"
            f"-------------\n"
        )

        candidate_fsp = list(current_fsp)

        proposed_flips = 0
        local_attempts = 0
        max_local_attempts = 50

        while (
            proposed_flips < k
            and local_attempts < max_local_attempts
        ):
            local_attempts += 1

            p = random.randrange(P)
            base_row = candidate_fsp[p]

            flips_here = random.randint(
                1,
                max(1, min(k - proposed_flips, J1 - 1)),
            )

            cand_row = random_flip_structure(
                J1,
                base_row,
                flips_here,
            )

            coeff = get_coeff_for_structure(cand_row)

            # Fallback: use an accurate structure already known.
            if coeff is None and trash_valid_fsp["coeff"]:
                valid_pool = list(
                    trash_valid_fsp["coeff"].keys()
                )
                random.shuffle(valid_pool)

                replaced = False

                for known_row in valid_pool:
                    other_rows = [
                        row
                        for index, row in enumerate(candidate_fsp)
                        if index != p
                    ]

                    if known_row in other_rows:
                        continue

                    candidate_fsp[p] = known_row
                    replaced = True
                    break

                if replaced:
                    proposed_flips += flips_here

                continue

            if coeff is None:
                continue

            candidate_fsp[p] = cand_row
            proposed_flips += flips_here

        candidate_fsp = tuple(candidate_fsp)
        candidate_key = canonical_fsp(candidate_fsp)

        if candidate_key in FSP_store["valid"]:
            continue

        if candidate_key in FSP_store["trash"]:
            continue

        true_dist = count_differences(
            current_fsp,
            candidate_fsp,
        )

        if true_dist < k:
            continue

        candidate_data = evaluate_candidate(candidate_fsp)

        if candidate_data is not None:
            return unpack_candidate(candidate_data)


    return (
                None,
                None,
                V,
                W,
                E,
                T,
                trash_valid_fsp,
                FSP_store,
            )

   
        
        

# Perturbation for dsa
def perturbation_dsa_(
    GLM, dataset,target,features,P,J1,k,tau,theta,gamma,SQ,
    FSP,FSP_store,trash_valid_fsp,
    ):
    ###########################################################################
    # Initialization
    ###########################################################################
    eps = 1e-8

    FSQ = [
        tuple(
            0 if abs(value) < eps else 1
            for value in SQ.iloc[q]
        )
        for q in range(len(SQ))
        ]

    coeff_cache = {}

    ###########################################################################
    # Helpers
    ###########################################################################
    def hamming_distance(row1, row2):
        return sum( bit1 != bit2 for bit1, bit2 in zip(row1, row2) )

    def row_ok_wrt_SQ(row):
        """
        A row is admissible with respect to SQ when its Hamming
        distance from every SQ structure is at least gamma.
        """
        return all(
            hamming_distance(row, fsq_row) >= gamma for fsq_row in FSQ
            )

    def satisfies_gamma_constraint(fsp_star):
        """
        Check both outer and inner DSA dispersion.
        """

        # Outer dispersion with respect to SQ
        for star_row in fsp_star:
            if not row_ok_wrt_SQ(star_row):
                return False

        # Inner dispersion between the P structures
        for first in range(len(fsp_star)):
            for second in range(first + 1, len(fsp_star)):
                if ( hamming_distance( fsp_star[first], fsp_star[second]) < gamma):
                    return False

        return True

    def get_coeff_for_structure(fsp_row):
        """
        Check the accuracy requirement of one structure and cache
        its maximum-accuracy coefficients.
        """
        row = tuple(fsp_row)

        if row in trash_valid_fsp["trash"]:
            return None

        if sum(row[1:]) > theta:
            trash_valid_fsp["trash"].add(row)
            return None

        if not row_ok_wrt_SQ(row):
            trash_valid_fsp["trash"].add(row)
            return None

        if row in coeff_cache:
            return coeff_cache[row]

        if row in trash_valid_fsp["coeff"]:
            coeff = trash_valid_fsp["coeff"][row]
            coeff_cache[row] = coeff
            return coeff

        accurate, coeff = acc_requirements(
            GLM,dataset,target,features,list(row),J1,tau,
            )

        if not accurate:
            trash_valid_fsp["trash"].add(row)
            return None

        trash_valid_fsp["coeff"][row] = coeff
        coeff_cache[row] = coeff

        return coeff

    def evaluate_candidate(candidate_fsp):
        """
        Check and store an entire candidate containing P structures.
        """
        candidate_fsp = tuple(
            tuple(row)
            for row in candidate_fsp
        )
        candidate_key = canonical_fsp(candidate_fsp)

        if candidate_key in FSP_store["trash"]:
            return None

        if candidate_key in FSP_store["valid"]:
            return None

        if not satisfies_gamma_constraint(candidate_fsp):
            FSP_store["trash"].add(candidate_key)
            return None

        coefficients = []

        for row in candidate_fsp:
            coeff = get_coeff_for_structure(row)

            if coeff is None:
                FSP_store["trash"].add(candidate_key)
                return None

            coefficients.append(coeff)

        candidate_data = {
            "fsp": candidate_fsp,
            "coeff": coefficients,
        }

        FSP_store["valid"][candidate_key] = candidate_data
        return candidate_data

    def unpack_candidate(candidate_data):
        return (
            candidate_data["fsp"],
            candidate_data["coeff"],
            trash_valid_fsp,
            FSP_store,
        )

    ###########################################################################
    # Initial solution
    ###########################################################################
    def build_initial_solution():
        selected_rows = []
        failed_extensions = 0
        attempts = 0
        max_attempts_init = 2500

        while attempts < max_attempts_init:
            attempts += 1

            print(
                f"________Build solution attempt "
                f"-- {attempts} -- __________"
            )

            valid_known_rows = [
                row
                for row in trash_valid_fsp["coeff"]
                if row_ok_wrt_SQ(row)
            ]

            use_known = (
                bool(valid_known_rows)
                and random.random() < 0.8
            )

            if use_known:
                candidate_row = random.choice(valid_known_rows)
            else:
                candidate_row = generate_valid_list(J1, theta)

            candidate_row = tuple(candidate_row)
            coeff = get_coeff_for_structure(candidate_row)

            if coeff is None:
                continue

            if candidate_row in selected_rows:
                continue

            trial = selected_rows + [candidate_row]

            if satisfies_gamma_constraint(trial):
                selected_rows.append(candidate_row)
                failed_extensions = 0
            else:
                failed_extensions += 1

                if failed_extensions >= 5:
                    selected_rows = []
                    failed_extensions = 0

            if len(selected_rows) < P:
                continue

            candidate_data = evaluate_candidate(
                tuple(selected_rows)
            )

            if candidate_data is not None:
                return candidate_data

            selected_rows = []

        return None

    ###########################################################################
    # Complete a partial solution using known accurate rows
    ###########################################################################
    def complete_partial_solution(partial_rows):
        selected_rows = list(partial_rows)

        valid_pool = [
            row
            for row in trash_valid_fsp["coeff"]
            if (
                row not in selected_rows
                and row_ok_wrt_SQ(row)
            )
        ]

        valid_pool.sort(
            key=lambda row: min(
                (
                    hamming_distance(row, fsq_row)
                    for fsq_row in FSQ
                ),
                default=float("inf"),
            ),
            reverse=True,
        )

        for candidate_row in valid_pool:
            if len(selected_rows) >= P:
                break

            trial = selected_rows + [candidate_row]

            if satisfies_gamma_constraint(trial):
                selected_rows.append(candidate_row)

        if len(selected_rows) != P:
            return None

        return tuple(selected_rows)

    ###########################################################################
    # CASE 1: no incumbent
    ###########################################################################
    if not FSP:
        candidate_data = build_initial_solution()

        if candidate_data is not None:
            return unpack_candidate(candidate_data)

        return (None,None,trash_valid_fsp,FSP_store)

    ###########################################################################
    # CASE 2: perturb the incumbent
    ###########################################################################
    current_fsp = tuple(tuple(row) for row in FSP)

    # Retain only incumbent rows compatible with SQ.
    valid_current_rows = tuple(row for row in current_fsp if row_ok_wrt_SQ(row))

    if len(valid_current_rows) == 0:
        candidate_data = build_initial_solution()

        if candidate_data is not None:
            return unpack_candidate(candidate_data)

        return (
            None,
            None,
            trash_valid_fsp,
            FSP_store,
        )

    if len(valid_current_rows) < P:
        completed_fsp = complete_partial_solution(
            valid_current_rows
        )

        if completed_fsp is not None:
            current_fsp = completed_fsp
        else:
            candidate_data = build_initial_solution()

            if candidate_data is not None:
                return unpack_candidate(candidate_data)

            return (
                None,
                None,
                trash_valid_fsp,
                FSP_store,
            )
    else:
        current_fsp = valid_current_rows

    max_attempts = 150

    for attempt in range(max_attempts):
        print(
            f"\n-------------\n"
            f" attempt : {attempt + 1}\n"
            f"-------------\n"
        )

        candidate_fsp = list(current_fsp)

        proposed_flips = 0
        local_attempts = 0
        max_local_attempts = 50

        while (
            proposed_flips < k
            and local_attempts < max_local_attempts
        ):
            local_attempts += 1

            p = random.randrange(P)
            base_row = candidate_fsp[p]

            flips_here = random.randint(
                1,
                max( 1, min( k - proposed_flips, J1 - 1) ),
            )

            candidate_row = random_flip_structure( J1, base_row, flips_here)

            coeff = get_coeff_for_structure(candidate_row)

            ###################################################################
            # Fallback: use a known accurate row
            ###################################################################
            if coeff is None and trash_valid_fsp["coeff"]:
                valid_pool = [
                    known_row
                    for known_row in trash_valid_fsp["coeff"]
                    if (
                        known_row != base_row
                        and row_ok_wrt_SQ(known_row)
                    )
                ]

                valid_pool.sort(
                    key=lambda known_row: (
                        hamming_distance(
                            known_row,
                            base_row,
                        ),
                        min(
                            (
                                hamming_distance( known_row, fsq_row)
                                for fsq_row in FSQ
                            ),
                            default=float("inf"),
                        ),
                    ),
                    reverse=True,
                )

                replaced = False

                for known_row in valid_pool:
                    trial_fsp = candidate_fsp.copy()
                    trial_fsp[p] = known_row

                    if satisfies_gamma_constraint(trial_fsp):
                        candidate_fsp[p] = known_row
                        replaced = True
                        break

                if replaced:
                    proposed_flips += flips_here

                continue

            if coeff is None:
                continue

            trial_fsp = candidate_fsp.copy()
            trial_fsp[p] = tuple(candidate_row)

            if not satisfies_gamma_constraint(trial_fsp):
                continue

            candidate_fsp[p] = tuple(candidate_row)
            proposed_flips += flips_here

        candidate_fsp = tuple(candidate_fsp)
        candidate_key = canonical_fsp(candidate_fsp)

        if candidate_key in FSP_store["valid"]:
            continue

        if candidate_key in FSP_store["trash"]:
            continue

        if not satisfies_gamma_constraint(candidate_fsp):
            continue

        true_distance = count_differences( current_fsp, candidate_fsp)

        if true_distance < k:
            continue

        candidate_data = evaluate_candidate(candidate_fsp)

        if candidate_data is not None:
            return unpack_candidate(candidate_data)

    ###########################################################################
    # No candidate found
    ###########################################################################
    return (
        None,
        None,
        trash_valid_fsp,
        FSP_store,
    )

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
