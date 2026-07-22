
"""
@author: Marica Magagnini

This file contains function to display results and evolutions of the optimization 
processes.
"""
from pyomo import environ as pym

###############################################################################
# Solver display
###############################################################################

# This function print and return the value of the obj function and the current solution
def get_obj_sol_(m):
       
    # Print new solution 
    print("The current set of P new models is: ")
    betaP_star = {}
    for p in m.p:
        print('Model %d)' % (p+1))

        betaP_star_p= {}
        for f in m.j_b0:
            betaP_star_p[f] = m.beta[p,f].value
            print(f"beta[{p},{f}] = {m.beta[p,f].value}") 
           
        print("\n")
        betaP_star[p] = betaP_star_p # New solution 
    
    
    for obj in m.component_objects(pym.Objective, active=True):
        obj_star = pym.value(obj)
        print(f"Current obj value: {obj_star}")


    return obj_star, betaP_star


###############################################################################
# VNS display
###############################################################################

# This function prints the VNS process from the iterations point of view. 
def printIterations(w, fail, new, worse):
    print("\nNumber of while iterations: %d" % w) 
    print('Number of times the perturbation step failed finding a feasible cominatorial framework: %d' % fail)
    print('Number of new local solution found: %d' %new)
    print('Number of new models sets worse than already existing ones: %d' % worse)
    return


# This fuction display the overall heuristic performance.
def print_VNS_RunResults( FSP, betaP, obj,
                              betaP_evolution, obj_evolution, 
                              t,Ts, Tf, TimeLimit,T_Heur_find,
                              w, fail, new, worse,
                              dataset, P):

    print("\n------ Heuristic Results -------\n")
    #Number of improvements found
    c = len(betaP_evolution)
    
    
    if c > 0:
        print(str(c)+ " Better betaP sets have been found (including the starter).")
        print('The heuristic solution is:  ')
        for BP,p in zip(betaP, range(P)):
            print(f'beta_{p} : ')
            print(BP)

        print("The objective function value is f = "+str(obj))
        
        if t >= TimeLimit:
            print("Time limit of " + str(TimeLimit)+ " seconds reached.")
        else:
                print("Total execution time  = %f seconds " % (Tf-Ts))
        
        print("\n------ Evolutions -------\n")
        print("Combinations explored with improvements: ")
        for BP, i in zip(betaP_evolution, range(1,len(betaP_evolution)+1)):
            for BP_p,p in zip(BP, range(P)):
                print(f'beta_{p+1} : ')    
                print(BP_p)
            print("f_obj_"+str(i)+ " = " + str (obj_evolution[i-1]))
            
        print("\nBest solution found after %s seconds." % T_Heur_find)

    else:
        print("Heuristic failed.")


    RES = {'Best solution (BetaP)' : betaP, 'FSP_best' : str(FSP),
           'obj_best': str(obj), 'Time fc found (s)': T_Heur_find, 
           '# while iter' : w, '# Faults' : fail, '# feasible sol' : new, 
           '# feasible count. worse' : worse, '# improvements' : (new - worse)}
    printIterations(w, fail, new, worse)
    
    return RES
