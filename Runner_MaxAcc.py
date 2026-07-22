
"""

@author: Marica Magagnini

This is the main file to run the P-CD problem maximizing the accuracy

"""
User = '..' 
folder = '..' 
from Funs import Dataset_selection, S_Q_, params_, PCD_problem_MAXACC_, save_results_coeff_xlsx, save_objRes_xlsx
import os
import pandas as pd
import numpy as np


# 
# GML : Linear Regression (Lin), Logistic Regression (Log), Poisson Regression (Poi)
#
GLM_list = ['Lin','Log','Poi']
name_dataset ='BH' #'ITT' BH
#
# Dispersion Notion: 'l1', 'l2', 'dsa', 'o1'   
# 
Dispersion_list = ['l1', 'l2', 'dsa', 'o1']


#
# Solver optimal or local
#
Solvers = ['gurobi_persistent', 'ipopt']
Solver_ = Solvers[1]

#
# P-CD_GLM Parameters
#
P = 3                          # Number of new GLMs
M = 200                        # big M

OPT_comparison = False
(gap_tol, obj_opt) = None, None


# Dispersion percentage gamma = g_pc*gamma_max with g_pc in GAMMA_percentage
GAMMA_percentage_disp = {'dsa': [1,2,3], '-':[.85, .75, .65]}#

THETA  = [20,23,25] if  name_dataset == 'ITT' else [6,7,8,9]
#%% Function to retrieve a feasible starting solution
def feas_sol(tau, theta, g_pc, gamma, dispersion, pool, THETA, GAMMA_percentage_disp):
    """

    Provide an initial feasible solution for the instance (tau, theta, g_pc)
    using previously solved instances contained in `pool`.

    Parameters
    ----------
    tau : float
        Accuracy threshold.
    theta : int | float
        Current value of the parameter theta. Must belong to THETA.
    g_pc : float | int
        Current level of dispersion. Must belong to
        GAMMA_percentage_disp[dispersion].
    gamma : float
        Value of gamma required by the current instance.
    dispersion : str
        Type of dispersion, for example 'dsa' or '-'.
    pool : dict
        Dictionary of previously known results:
            pool[(tau, theta, g_pc)] = {
                "gamma": gamma,
                "obj": obj,
                "SP": SP
            }
    THETA : list
        Ordered list of allowed values for theta.
    GAMMA_percentage_disp : dict
        Dictionary of allowed values for g_pc for each type of dispersion.
        Example:
            {
                'dsa': [1, 2, 3],
                '-': [0.85, 0.75, 0.65]
            }

    Returns
    -------
    SP_feas : list
        Initial feasible solution.
    obj_feas : float
        Objective value associated with the solution.
    """

    G_PC = GAMMA_percentage_disp[dispersion] if dispersion == 'dsa' else GAMMA_percentage_disp['-']
    

    if theta not in THETA:
        raise ValueError(f"theta={theta} non appartiene a THETA={THETA}")
    if g_pc not in G_PC:
        raise ValueError(
            f"g_pc={g_pc} non appartiene a GAMMA_percentage_disp['{dispersion}']={G_PC}"
        )

    theta_idx = THETA.index(theta)
    g_idx = G_PC.index(g_pc)

    def build_g_pc_feas(dispersion, g_idx, G_PC):
        """
        Restituisce il g_pc adiacente da cui ereditare una soluzione ammissibile.

        Logica coerente con la funzione originale:
        - per 'dsa': si prende il valore precedente nel vettore
        - per gli altri casi: si prende il valore precedente nel vettore,
          che nel tuo esempio corrisponde a passare da 0.75 -> 0.85,
          0.65 -> 0.75, ecc.
        """
        if g_idx == 0:
            return None
        return G_PC[g_idx - 1]

    SP_feas, obj_feas = [], 1e10

    is_first_theta = (theta_idx == 0)
    is_first_g_pc = (g_idx == 0)

    # Caso base: nessuna soluzione nota per il primo theta e il primo g_pc
    if is_first_theta and is_first_g_pc:
        print('Base case: no feas solution available')
        pass

    # Primo theta, ma non primo g_pc:
    # uso il g_pc precedente nel vettore della dispersione corrente
    elif is_first_theta and not is_first_g_pc:
        g_pc_feas = build_g_pc_feas(dispersion, g_idx, G_PC)
        print('fesible (theta, g_pc) = ', (theta_idx,g_pc_feas))
        if g_pc_feas is not None:
            result = pool.get((tau, theta, g_pc_feas))
            if result is not None:
                SP_feas = result.get("SP", [])
                obj_feas = result.get("obj", 1e10)
            else:
                print("No feasible solution available.")
        else:
            print("No feasible solution available.")

    # Theta non primo, g_pc primo:
    # provo a ereditare da theta precedente a parità di g_pc
    elif not is_first_theta and is_first_g_pc:
        theta_feas = THETA[theta_idx - 1]
        result = pool.get((tau, theta_feas, g_pc))
        print('fesible (theta, g_pc) = ', (theta_feas,g_idx))
        if result is not None:
            # gamma(theta) <= gamma(theta precedente)
            if gamma <= result.get("gamma", float("-inf")):
                SP_feas = result.get("SP", [])
                obj_feas = result.get("obj", 1e10)
        else:
            print("No feasible solution available.")

    # Caso generale: theta non primo e g_pc non primo
    else:
        # Cerco tra tutte le soluzioni note con lo stesso tau
        # una che soddisfi la soglia gamma e minimizzi obj
        for key, value in pool.items():
            tau_i, theta_i, g_pc_i = key
            if tau_i == tau and value.get("gamma", float("-inf")) >= gamma and theta_i <= theta:
                if value.get("obj", 1e10) <= obj_feas:
                    obj_feas = value["obj"]
                    SP_feas = value["SP"]

        if SP_feas == []:
            print("No feasible solution available.")
            
            
            

    return SP_feas, obj_feas


def get_obj_opt_(filename, directory, tau, theta, gamma):
    """
    Reads a file (csv or excel), filters by the first 3 columns
    and returns the value of the fourth column.

    Parameters:
        filename (str): name of the file
        directory (str): path of the directory
        val1, val2, val3: values to search in the first 3 columns
    
    Returns:
        value of the 4th column or None if not found
    """
    
    path = os.path.join(directory, filename)
    
    # Reading file
    if filename.endswith('.csv'):
        df = pd.read_csv(path)
    elif filename.endswith('.xlsx') or filename.endswith('.xls'):
        df = pd.read_excel(path)
    else:
        raise ValueError("Format not supported")
    
    # Column check
    if df.shape[1] < 4:
        raise ValueError("The file must have at least 4 columns")
    
    # Filtro
    result = df[
        (df.iloc[:, 0] == tau) &
        (df.iloc[:, 1] == theta) &
        (df.iloc[:, 2] == gamma)
    ]
    
    if not result.empty:
        return result.iloc[0, 3]   # fourth column
    else:
        return None
#%% Executions


runs_vns = 1 # Change for a Multi-start approach for the heuristic


for GLM , Solver_ in [ ('Lin','gurobi_persistent'), ('Lin','ipopt'),('Log','ipopt'),('Poi','ipopt')]:
    #
    # Data    
    #
    dataset, target, features, features_type = Dataset_selection(GLM, name_dataset)
    N,J = dataset.shape


    X0 = dataset.iloc[0]
    X0["bias"] = 1
    X0 = X0[["bias"] + [col for col in X0.index if col != "bias"]]

    #
    # Input model
    #
    SQ, SQ_obj = S_Q_(GLM, features, name_dataset)  
    

    TimeLimit_vns =3600
        
    for dispersion in Dispersion_list:
        if Solver_ == 'gurobi_persistent':
            logfile = open(f"A_log_solver_status_{GLM}_{dispersion}.txt", "a")
        else:
            logfile = open(f"A_heur_solver_status_{GLM}_{dispersion}.txt", "a")
        
        print('---------------------------------------------------------------------------')
        print(f'GLM: {GLM}, dispersion: {dispersion}. N = {N}, J = {J}')
        print('Parameters:')
        #
        # Parameters
        #
        parameters = params_(GLM, dispersion)

        #Indexed by t_pc, theta, g_pc, contains gamma, SP_obj and SP
        pool = {}
        
      
       
        objs = []
        for par in parameters.values: 

            tau, theta, gamma_max =( par.item(0), int(par[1]), par.item(2))

            #
            # Define gamma iterations
            #
            if gamma_max == 0  :
                GAMMA_percentage = []
            elif dispersion == 'dsa':
                GAMMA_percentage = GAMMA_percentage_disp['dsa']
            else:
                GAMMA_percentage = GAMMA_percentage_disp['-']
              

            for g_pc  in GAMMA_percentage :
                if dispersion == 'dsa':
                    gamma = int(gamma_max - g_pc)
                    gamma = max(0, gamma)
                    print(f' tau :{tau}, theta: {theta},  gamma: {gamma} (gamma_max {gamma_max})')
            
                else:
                    gamma  = round(gamma_max * g_pc,3)
                    print(f' tau :{tau}, theta: {theta},  gamma: {gamma} ({g_pc*100}% gamma_max ({round(gamma_max,3)}))')
                
                # Feasible solution 

                
                SP_feas, obj_feas = feas_sol(
                                                tau=tau,
                                                theta=theta,
                                                g_pc=g_pc,
                                                gamma=gamma,
                                                dispersion=dispersion,
                                                pool=pool,
                                                THETA=THETA,
                                                GAMMA_percentage_disp=GAMMA_percentage_disp
                                            )
                #
                # If opt sol available
                #                                                
                if OPT_comparison:
                    filename = f"A_obj_{GLM}_{dispersion}.xlsx"
                    directory = f'C://Users//...//A//OPT - Lin//{dispersion}'
                    obj_opt = get_obj_opt_(filename, directory, tau, theta, gamma)
                    gap_tol = 1e-5
                    print(f'Optimal reference - ({tau}, {theta}, {gamma}) : {obj_opt}')
                #                                            
                # Execution
                #   
                SP, SP_obj, status = PCD_problem_MAXACC_(GLM, Solver_, 
                                          dataset,target,features, 
                                          
                                          SQ, P, gamma, tau, theta, M, X0,
                                          dispersion,  runs_vns,TimeLimit_vns,
                                          SP_feas, obj_feas,
                                          obj_opt= obj_opt, gap_tol=gap_tol)
                
                if SP != []:
                    # Save Results
                    save_results_coeff_xlsx('A',Solver_, GLM, dispersion, theta,tau, gamma, 
                                                SP,SP_obj, SQ, SQ_obj,  features, filename = None)
                
                objs.append((tau, theta, gamma,  SP_obj))
                
                # Record all gamma, SP and SP_obj associated to tau, theta and g_pc 
                pool[(tau,theta, g_pc)] = {
                    "gamma" : gamma,
                    "obj" : SP_obj,
                    "SP" : SP}
                
                #
                #  Write on log -- Solver/heuristic status
                #
                logfile.write(f"GLM={GLM}, Dispersion={dispersion}, tau={tau:.5f}, theta={theta}, gamma={gamma} ")
                if Solver_ == 'gurobi_persistent':
                    logfile.write(f"Status: {status[2]}, ObjVal: {status[0]:.5f}, MIPGap: {status[1]:.5f}\n")
                else:
                    logfile.write(f'Status: {status}\n')
                
            
        save_objRes_xlsx('A',Solver_,GLM,dispersion, objs, filename = None)
        
        #close gurobi log file
        logfile.close()
