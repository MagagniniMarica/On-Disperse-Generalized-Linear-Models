"""
@author: Marica Magagnini

This script contains the VNS strategy to handle the combinatorial part for the 
P-CD problem maximizing the dispersion 
"""

import time
from f_print import  print_VNS_RunResults, get_obj_sol_
from P_CD_problem_MaxDisp.PCD_MAX_dispersion_VNS_ import P_max_disp_
from pyomo import environ as pym
from pyomo.opt import TerminationCondition
import numpy as np

from P_CD_problem_MaxDisp.funs_vns_Disp import update_k_,rel_gap_,  perturbation_,perturbation_dsa_, min_hamming_distance


# theta : maximal number of non-zero features 
#GLM : Generalized Linear Model -->'LinearRegression', 'LogisticRegression', 'PoissonRegression'
# dispersion : 'l2','dsa','l1','o1'
def VNS(GLM, Solver_, K,TimeLimit, 
        dataset, target, features, SQ, P,J1,theta, tau,dispersion, X0,
        solver_Iterations,  obj_feas, obj_ref=None, gap_tol=None):
    ###########################################################################
    # INITIALIZATION
    ###########################################################################
    
    t = 0                                               # Actual time
    Ts = time.time()                                    # Start Time
    T_Heur_find = 0                                     # Time best heuristic solution found

    # Store in these list the improvements found 
    FSP_evolution = []
    betaP_evolution = []
    obj_evolution = []
    
    obj_time_evolution = [{'time': 0,'obj': obj_feas}]
    
    k=1
    obj = obj_feas #Initial maximal dispersion
    FSP = []
    betaP = []
    trash_valid_fsp = {'trash': set(), 'coeff': {}} # set of trash and valid stuctures that do not and do respect the accuracy
    FSP_store =  set()
    ###########################################################################
    #Some index to understand the performances
    ##
    w = 0                       # Number of while iteration
    failed = 0                  # Number of times the perturbation failed
    new = 0                     # Number of new local solutions betaP set
    worse = 0                   # Number of betaP sets worse than already existing ones
    # If optimal solution avalible
    rel_gap =  rel_gap_(obj, obj_ref, gap_tol )
    ##
    ###########################################################################
    
    ###########################################################################
    # VNS CYCLE
    ###########################################################################
    
    while (k < K) and (t < TimeLimit) and ((rel_gap is None) or (rel_gap > gap_tol)):
        w +=1
        #######################################################################
        # Perturbation
        #######################################################################
        print('Perturbation')
        
        """
        if dispersion == 'o1':
            V,W =  o1_dispersion_(X0, SQ, P) # combinatorial structure of l1 norm on the outputs
        if dispersion == 'l1':
            E,T =  l1_dispersion_(P,len(SQ),features) # combinatorial structure of l1 norm 
            
        Note: the perturbation function yields a set of model with the minimal accuracy requirement.
        The coefficients of these models are used to feed the main local problem with starting solution.
        """
        if dispersion == 'dsa':
            FSP_star, FSP_star_coeff, trash_valid_fsp, FSP_store =  perturbation_dsa_(GLM, dataset, target,
                                                                                      features,P,J1,k,tau, 
                                                                                      theta, obj,SQ,
                                                                                      FSP,FSP_store,
                                                                                      trash_valid_fsp )
            

            (V,W,T,E) = (None, None, None, None)
        else: 
            FSP_star, FSP_star_coeff, V,W,E,T, trash_valid_fsp,FSP_store = perturbation_(FSP_store, FSP,
                                                                                      theta, k, P,
                                                                                      GLM, dataset, target, features, 
                                                                                      tau, J1,
                                                                                      dispersion, SQ, X0,
                                                                                      solver_Iterations,
                                                                                      trash_valid_fsp)


        print('Done')

        
        #######################################################################   
        # Local Problem - Max min disp
        #######################################################################
            
        ts = time.time()
       
        if FSP_star:
        
            if dispersion == 'dsa':
               # check actual dispersion
               obj_star = min_hamming_distance(FSP_star, SQ) 
               (FSP_star, betaP_star) = FSP_star, FSP_star_coeff
               new = new +1            # counter new local solution incremented
               
               
            else: 
               
                instance  =  P_max_disp_(GLM,dispersion, dataset, target, features, FSP_star,
                                         SQ, P, tau, X0, V = V,W = W, T=T, E=E)
                
                #
                # === Provide Initial Guess for Ipopt === the one with maximal accuracy
                #
                for p in range(P):
                    for f in instance.j_b0:     
                        instance.beta[p, f].value = FSP_star_coeff[p][f]  # or some heuristic/prior guess
            
            
                # === Solve with Ipopt ===
                solver = pym.SolverFactory(Solver_)
                
                solver.options['print_level'] = 5               # Controls verbosity (0 = silent, 5 = detailed)
                solver.options['max_iter'] = solver_Iterations  # Max iterations
                solver.options['tol'] = 1e-6                    # Convergence tolerance
                solver.options['acceptable_tol'] = 1e-5
                
                try:
                    results = solver.solve(instance, tee=True)
    
                    term_cond  = results.solver.termination_condition
                
                except Exception as e:
                    term_cond = None
                    print(f"[ERROR] Error during the resolution with Ipopt: {e}")
        
        
                
                te = time.time()
        
                print(f'Execution time : {te-ts} s')
               
            
                #
                # Solver Termination Condition
                #
                if term_cond in [TerminationCondition.optimal, TerminationCondition.feasible] :            # optimal or accepted tollerances
                    print("Ipopt found a solution:", term_cond)    
                    new = new +1            # counter new local solution incremented
                    obj_star, betaP_star = get_obj_sol_(instance)
                else:
                    k = update_k_(k,K)
                    print("Local Solution not found.")
                    continue
                
            ###############################################################
            # Check if betaP_star is better than the current one
            ###############################################################
            if (obj_star > obj) :      
                print('\n ----------\n ----------\n IMPROVEMENT \n ----------\n ----------\n')
            
                # Update current best heuristic solution 
                (FSP,obj, betaP) = (FSP_star,obj_star, betaP_star)
                
                # store the solution
                FSP_evolution.append(FSP)
                betaP_evolution.append(betaP)
                obj_evolution.append(obj)
                
                # VNS restart
                k = 1
                # Time betaP_star was found
                T_Heur_find = time.time() - Ts   
                obj_time_evolution.append({
                                            'time': T_Heur_find,
                                            'obj': obj
                                            })
                rel_gap =  rel_gap_(obj, obj_ref, gap_tol )        # Update relative gap
                                       
            
            else:                                                       # betaP_star worse than currentbetaP
                k = update_k_(k,K)
                print("Found a new solution WORSE than the current one.")
                worse += 1 
        
            

        else:
            k = update_k_(k,K)
            failed +=1
            print("Perturbation failed.")     
        
        # Time one iteration 
        t = time.time() - Ts                                                # current time    
        print('------------------ Next Loop Iteration --------------------')
        
    Tf = time.time()                                                        # Final Time
                        
    ###############################################################################      
    # Print Heuristic Run performance and assign results RES
    ###############################################################################    
    RES  = print_VNS_RunResults( FSP, betaP, obj,
                                  betaP_evolution, obj_evolution, 
                                  t,Ts, Tf, TimeLimit,T_Heur_find,
                                  w, failed , new, worse,
                                  dataset, P)
    
    return RES, FSP, betaP, obj, obj_time_evolution


              
                
            
                
        
    