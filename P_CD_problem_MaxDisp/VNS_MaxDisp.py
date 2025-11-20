# -*- coding: utf-8 -*-
"""
@author: Marica Magagnini

This script contains the VNS strategy to handle the combinatorial part for the 
problem of maximizing the dispersion.
"""

import time
from f_print import  print_VNS_RunResults, get_obj_sol_
from P_CD_problem_MaxDisp.PCD_MAX_dispersion_VNS_ import P_max_disp_local_
from pyomo import environ as pym
from pyomo.opt import TerminationCondition


from P_CD_problem_MaxDisp.funs_vns_Disp import update_k_,  perturbation_,perturbation_dsa_, min_hamming_distance, Hamming_epsilon


# theta : maximal number of non-zero features 
#GLM : Generalized Linear Model -->'LinearRegression', 'LogisticRegression', 'PoissonRegression'
# dispersion : 'l2','l1','o1'
def VNS(GLM, Solver_, K,TimeLimit, 
        dataset, target, features, SQ, P,J1,theta, tau,dispersion, X0,
        solver_Iterations):
    ###########################################################################
    # INITIALIZATION
    ###########################################################################
    
    t = 0                                               # Actual time
    Ts = time.time()                                    # Start Time
    T_Heur_find = 0                                     # Time best heuristic solution found

    # Track the evolutions
    FSP_evolution = []
    betaP_evolution = []
    obj_evolution = []

    k=1
    obj = 0 #Initial maximal dispersion
   
    FSP = []
    betaP = []
    trash_valid_fsp = {'trash': set(), 'valid': set()} # set of trash and valid stuctures that do not and do respect the accuracy
    FSP_store =  set()
    ###########################################################################
    #Some index to understand the performances
    ##
    w = 0                       # Number of while iteration
    failed = 0                  # Number of times the perturbation failed
    new = 0                     # Number of new local solutions betaP set
    worse = 0                   # Number of betaP sets worse than already existing ones
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
        
        
        FSP_star, FSP_star_coeff, V,W,E,T, trash_valid_fsp,FSP_store = perturbation_(FSP_store, FSP,
                                                          theta, k, P,
                                                          GLM, dataset, target, features, 
                                                          tau, J1,
                                                          dispersion, SQ, X0,
                                                          solver_Iterations,
                                                          trash_valid_fsp)


        print('Done')

        
        #######################################################################   
        # Local Problem - Max mininimal dispersion
        #######################################################################
            
        ts = time.time()
       
        if FSP_star:

            instance  =  P_max_disp_local_(GLM,dispersion, dataset, target, features, FSP_star,
                                     SQ, P, tau, X0, V = V,W = W, T=T, E=E)
            
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
                new = new +1            # counter new local solution incremented
                obj_star, betaP_star = get_obj_sol_(instance)
            
                
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
    
    return RES, FSP, betaP, obj



# theta : maximal number of non-zero features 
#GLM : Generalized Linear Model -->'LinearRegression', 'LogisticRegression', 'PoissonRegression'
# Dispersion :  Different Subsets of Attributes (Hamming distance)
def VNS_dsa_(GLM, Solver_, K,TimeLimit,
        dataset, target, features, SQ, P,J1,theta, tau, X0,
        solver_Iterations):
   
    ###########################################################################
    # INITIALIZATION
    ###########################################################################
    Ts = time.time()                                    # Start Time
    T_Heur_find = 0                                     # Time best heuristic solution found


    # Store in these list the improvements found 
    # but the VNS during the run
    FSP_evolution = []
    betaP_evolution = []
    obj_evolution = []

    betaP = []
    obj = 0 #Initial maximal dispersion
    FSP = []
    trash_valid_fsp = {'trash': set(), 'valid': set()} # set of trash and valid stuctures that do not and do respect the accuracy
    FSP_store = set()
    ###########################################################################
    

    gamma = 0
    
    
    while gamma < J1:
        
        
        ###########################################################################
        #  A VNS CYCLE for each gamma
        ###########################################################################
        ###########################################################################
        #Some index to understand the performances
        ##
        w = 0                       # Number of while iteration
        failed = 0                  # Number of times the perturbation compute an FSP already used
        new = 0                     # Number of new local solutions betaP set
        worse = 0                   # Number of betaP sets worse than already existing ones
        ##
        t = 0                       # Actual time
        Tsg = time.time()           # Start Time gamma
        ###########################################################################
        
        k = 1 # vns index
        
        
        
        while (k < K) and (t < TimeLimit):
            w +=1
            #######################################################################
            # Perturbation
            #######################################################################
            print('Perturbation')
            FSP_star, FSP_star_coeff,trash_valid_fsp, FSP_store = perturbation_dsa_(GLM, dataset, target, 
                                                         features,P,J1,k,
                                                         tau, theta, gamma,
                                                         SQ,FSP,FSP_store,
                                                         trash_valid_fsp)
            
            
            ###################################################################
            # Check if a feasibles cobination has been found
            ###################################################################
            if FSP_star:
                
                # Overall Hamming dispersion between B (FSP_star), considering B_0 (SQ)
                betaP_star_HD = min_hamming_distance(FSP_star, SQ) 
                
                # Check if the new dispersion is better and the hamming_epsilon dispersion definition is satisfied
                if betaP_star_HD >= obj and Hamming_epsilon(FSP_star_coeff, 1e-8):
                    obj = betaP_star_HD               
                    betaP = FSP_star_coeff
                    FSP = FSP_star

                    
                    new = new +1
                    T_Heur_find = time.time()
                    
                    # store the solution
                    FSP_evolution.append(FSP_star)
                    betaP_evolution.append(betaP)
                    obj_evolution.append(obj)
                    
                    # If there is a solution B for the current gamma, let go to 
                    # if there exists one for gamma+1
                    break
                    
                else:                                                       # betaP_star worse than current betaP
                    print("Found a new solution WORSE than the current one.")
                    worse += 1 
                    k = update_k_(k,K)
            
              
            else:
                k = update_k_(k,K)
                failed +=1
                print("Perturbation failed.")
            
                
            t = time.time() - Tsg                                               # current time    
        
        print(f'------------------ (requested gamma  = {gamma}) Next Loop Iteration --------------------')
                  
        Tf = time.time()                                                        # Final Time
            
  

                    
        
        #######################################################################
        # Next gamma
        #######################################################################
        
        # If there exists a solution B for the current gamma, go to next gamma
        if new > 0:
            gamma = obj + 1 # next gamma is the current one +1
            print('-------------------------------------------------------')
            print(f'------------------ actual gamma  = {gamma} --------------------')
            print('-------------------------------------------------------')
        else:
            # Stop: no solution have been found in for current gamma
            # Best solution is the last one.
            print('-------------------------------------------------------')
            print(f'------------------ Best gamma = {obj} --------------------')
            print('-------------------------------------------------------')
            break
        
        
    ###############################################################################      
    # Print Heuristic Run performance and assign results RES
    ###############################################################################    

    RES =[]    
    RES  = print_VNS_RunResults( FSP, betaP, obj,
                                  betaP_evolution, obj_evolution, 
                                  t,Ts, Tf, TimeLimit,T_Heur_find,
                                  w, failed, new, worse,
                                  dataset, P)       
    
        
    return RES, FSP, betaP, obj

                 
                
            
                
        
    
