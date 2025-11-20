# -*- coding: utf-8 -*-
"""
@author: Marica Magagnini

This file containts the fuction of the VNS heuristic strategy to solve Problem (A-PDP-GLM).

"""

import time
from f_print import  print_VNS_RunResults, get_obj_sol_
from P_CD_problem_MaxAcc.PCD_Max_Acc_VNS import P_cond_disp_
from pyomo import environ as pym
from pyomo.opt import TerminationCondition

from P_CD_problem_MaxAcc.funs_vns import update_k_,  perturbation_dsa_, perturbation_, Hamming_epsilon



# theta : maximal number of non-zero features 
def VNS(GLM, Solver_, K,TimeLimit,
        dataset, target, features, SQ, P,J1,theta, gamma, tau,dispersion, X0,
        solver_Iterations):
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

                                                                   
    k=1                                                 # VNS index, i.e. number of changing activation of features
    obj = 1e10                                          # Initialize min loss
    betaP = []
    FSP = []                                            # Feature Selection structure
    trash_valid_fsp = {'trash': set(), 'valid': set()}  # Set of trash and valid stuctures that do not and do respect the accuracy
    FSP_store = {'trash': set(), 'valid': set()}
    ###########################################################################
    #Some index to understand the performances
    ##
    w = 0                                               # Number of while iteration
    failed = 0                                          # Number of times the perturbation failed
    new = 0                                             # Number of new local solutions betaP set
    worse = 0                                           # Number of betaP sets worse than already existing ones
    ##
    ###########################################################################
    
    ###########################################################################
    # VNS CYCLE
    ###########################################################################
    
    while (k < K) and (t < TimeLimit):
        w +=1
        #######################################################################
        # Perturbation
        #######################################################################
        print('Perturbation')
        
        # FSP_star_coeff : coefficients of the solution models of the max dispersion problem 
        # with accuracy constraints. 
        if dispersion == 'dsa':
            FSP_star, FSP_star_coeff,trash_valid_fsp, FSP_store = perturbation_dsa_(GLM, dataset, target, 
                                                         features,P,J1,k,
                                                         tau, theta, gamma,
                                                         SQ,FSP,FSP_store,
                                                         trash_valid_fsp)
            (V,W,T,E) = (None, None, None, None)
        else:
        
            FSP_star, FSP_star_coeff, V,W,E,T, trash_valid_fsp,FSP_store = perturbation_(FSP_store, FSP,
                                                              theta, k, P,
                                                              GLM, dataset, target, features, 
                                                              tau, J1,
                                                              dispersion, SQ, X0, gamma,
                                                              solver_Iterations,
                                                              trash_valid_fsp)

    
        print('Done')
        #######################################################################   
        # Local Problem - Max sum acc
        #######################################################################
            
        ts = time.time()
       
        if FSP_star:
            instance = P_cond_disp_(GLM,dispersion , dataset, target, features, FSP_star,
                                      SQ, P, gamma, tau, X0, V = V,W = W, T=T, E=E)
            
    
            
            #
            # === Provide Initial Guess for Ipopt === the one with maximal accuracy
            #
            for p in range(P):
                for f in instance.j_b0:     
                    instance.beta[p, f].value = FSP_star_coeff[p][f]  
        
        
            # === Solve with Ipopt ===
            solver = pym.SolverFactory(Solver_)
            
            solver.options['print_level'] = 5               # Controls verbosity (0 = silent, 5 = detailed)
            solver.options['max_iter'] = solver_Iterations  # Max iterations
            solver.options['tol'] = 1e-6                    # Convergence tolerance
   
            
            try:
                results = solver.solve(instance, tee=True)

                term_cond  = results.solver.termination_condition
            
            except Exception as e:
                term_cond = None
                print(f"[ERROR] Errore durante la risoluzione con Ipopt: {e}")
    
    
            
            te = time.time()
    
            print(f'Execution time : {te-ts} s')
            
            #
            # Solver Termination Condition
            #
            if term_cond in [TerminationCondition.optimal, TerminationCondition.feasible] :            # optimal or accepted tollerances
                print("Ipopt found a solution:", term_cond)    
                new += 1            # counter new local solution incremented
                obj_star, betaP_star = get_obj_sol_(instance)

                # Check the hamming-epsilon distance definition is satisfied, if not skip solution
                if dispersion == 'dsa' and not Hamming_epsilon(betaP_star,1e-8): 
                    new -= 1
                    obj_star = 1e15   
                ###############################################################
                # Check if betaP_star is better than the current one
                ###############################################################
                if (obj_star < obj) :      # IMPROVEMENT  
                    print('\n ----------\n ----------\n IMPROVEMENT \n ----------\n ----------\n')

    
                
                    # Update current best heuristic solution 
                    (FSP,obj, betaP) = (FSP_star,obj_star, betaP_star)
                    
                    
                    k = 1                                                   # VNS restart
                    
                    # store the solution
                    FSP_evolution.append(FSP_star)
                    betaP_evolution.append(betaP)
                    obj_evolution.append(obj)
    
                    T_Heur_find = time.time() - Ts                          # Time betaP_star was found
                
                else:                                                       # betaP_star worse than currentbetaP
                    k = update_k_(k,K)
                    print("Found a new solution WORSE than the current one.")
                    worse += 1  
            else:
                k = update_k_(k,K)
                print("Local Solution not found.")     
        else:
            k = update_k_(k,K)
            failed +=1
            print("Perturbation failed.")     
                
        t = time.time() - Ts                                                # current time    
        print('\n------------------ Next Loop Iteration --------------------\n')
    
    Tf = time.time()                                                        # Final Time
                        
    ###############################################################################      
    # Print Heuristic Run performance and assign results RES
    ###############################################################################    
    RES  = print_VNS_RunResults( FSP, betaP, obj,
                                  betaP_evolution, obj_evolution, 
                                  t,Ts, Tf, TimeLimit,T_Heur_find,
                                  w, failed, new, worse,
                                  dataset, P)
    
    return RES, FSP, betaP, obj




    
