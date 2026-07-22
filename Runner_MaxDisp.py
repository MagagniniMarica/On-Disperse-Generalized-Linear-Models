
"""
@author: Marica Magagnini

This is the main file to run the P-CD problem maximizing the dispersion. 
"""
import pandas as pd
import os
#
# PC directory selection
#
User = '..' 
folder = '..' 
#
# Functions to run the experiments
#
from Funs import Dataset_selection, S_Q_, PCD_problem_MAXDISP_, save_results_coeff_xlsx, save_objRes_xlsx
name_dataset = 'ITT' # ITT or 'CC' else BH


# 
# GML : Linear Regression (Lin), Logistic Regression (Log), Poisson Regression (Poi)
#
GLM_list = ['Lin','Log','Poi']



#
# Dispersion Notion: 'l1', 'l2', 'dsa', 'o1'   
# 
Dispersion_list = ['l1', 'l2', 'dsa', 'o1']


#
# Solver optimal or local ( Note that gurobi solves only the linear case)
#
Solvers = ['gurobi_persistent', 'ipopt'] 
Solver_ = Solvers[1]

#
# P-CD_GLM Parameters
#
P =3                                        # Number of new GLMs
M =1e6 if name_dataset =='CC' else 1e3      # big M for linear case 
OPT_comparison = False
Pareto_Front = False
(gap_tol, obj_opt) = None, None

TAU_percentage = [0.1, 0.15, 0.2]   # tau = tau + tau_pc*tau_MIN

#%% Funtion to retrive a feasible starting solution

def feas_sol(theta, t_pc, pool, THETA, TAU_percentage):
    """
    Provide an initial feasible solution for the instance (theta, t_pc)
    using previously solved instances contained in `pool`.


    Parameters
    ----------
    theta : int | float
        Current value of the parameter theta. Must belong to THETA.
    t_pc : float
        Current value of the parameter t_pc. Must belong to TAU_percentage.
    pool : dict
        Dictionary containing previously solved instances.
            pool[(theta, t_pc)] = {
                "gamma_max": gamma_max,
                "SP": SP
            }
    THETA : list
        Ordered list of allowed values for theta.
    TAU_percentage : list
        Ordered list of allowed values for t_pc.

    Returns
    -------
    SP_feas : list
       Initial fesible solution for the instance (theta, t_pc)
    gamma_max_feas : float
        Associated value to the initial feasible solution
    """

    if theta not in THETA:
        raise ValueError(f"theta={theta} not in  THETA={THETA}")
    if t_pc not in TAU_percentage:
        raise ValueError(f"t_pc={t_pc} not in TAU_percentage={TAU_percentage}")

    theta_idx = THETA.index(theta)
    tau_idx = TAU_percentage.index(t_pc)

    # Base-case: first theta and first t_pc, no previous solution available
    if theta_idx == 0 and tau_idx == 0:
        return [], 0

    theta_feas = None
    t_pc_feas = None

    # Case: first theta, but not first t_pc
    if theta_idx == 0 and tau_idx > 0:
        theta_feas = theta
        t_pc_feas = TAU_percentage[tau_idx - 1]

    # Case: first t_pc, but not first theta
    elif theta_idx > 0 and tau_idx == 0:
        theta_feas = THETA[theta_idx - 1]
        t_pc_feas = t_pc

    # General case: comparison between (previous theta, same t_pc)
    # and (same theta, previous t_pc)
    else:
        prev_theta_key = (THETA[theta_idx - 1], t_pc)
        prev_tau_key = (theta, TAU_percentage[tau_idx - 1])

        option_1 = pool.get(prev_theta_key, {}).get("gamma_max", 0)
        option_2 = pool.get(prev_tau_key, {}).get("gamma_max", 0)

        if option_1 > option_2:
            theta_feas = THETA[theta_idx - 1]
            t_pc_feas = t_pc
        else:
            theta_feas = theta
            t_pc_feas = TAU_percentage[tau_idx - 1]

    result = pool.get((theta_feas, t_pc_feas))
    if result is not None:
        return result.get("SP", []), result.get("gamma_max", 0)
    else:
        print("Initial feasible solution not found")
        return [], 0

def get_obj_opt_(filename, directory, tau, theta):
    """

    Reads a file (csv or excel), filters by the first 2 columns, and returns the value of the third column.

    Parameters:
        filename (str): name of the file
        directory (str):  directory path
        val1, val2, : values to search in the first 2 columns
    
    Returns:
        value of the 3rd column or None if not found
    """
    
    path = os.path.join(directory, filename)
    
    # Reading file
    if filename.endswith('.csv'):
        df = pd.read_csv(path)
    elif filename.endswith('.xlsx') or filename.endswith('.xls'):
        df = pd.read_excel(path)
    else:
        raise ValueError("File format not supported")
    
    # Controllo colonne
    if df.shape[1] < 3:
        raise ValueError("The file must have at least 3 columns")
    
    # Filtro
    result = df[
        (df.iloc[:, 0] == tau) &
        (df.iloc[:, 1] == theta) 
    ]
    
    if not result.empty:
        return result.iloc[0, 2]   # third column
    else:
        return None
#%% Executions



runs_vns = 1 # Change for a Multi-start approach for the heuristic

for GLM , Solver_ in [('Lin', Solvers[0]), ('Lin', Solvers[1]), ('Log', Solvers[1]), ('Poi', Solvers[1])]:    
    #
    # Data    
    #
    dataset, target, features, features_type = Dataset_selection(GLM,name = name_dataset)
    N,J = dataset.shape
    THETA  =  [15,25,35,45] if  name_dataset == 'ITT' else [70, 95, 120, 148] if name_dataset=='CC' else [6,7,8,9]

    X0 = dataset.iloc[0]
    X0["bias"] = 1
    X0 = X0[["bias"] + [col for col in X0.index if col != "bias"]]

    #
    # Input model
    #
    SQ, tau_min = S_Q_(GLM, features,name = name_dataset)  #mancano quelli per poisson
    
    TimeLimit_vns =3600 if name_dataset == 'CC' else 1800 
    
    for dispersion in Dispersion_list:
        
        if Solver_ == 'gurobi_persistent':
            logfile = open(f"D_log_solver_status_{dispersion}.txt", "a")
        else:
            logfile = open(f"D_heur_status_{dispersion}.txt", "a")
        
        print('---------------------------------------------------------------------------')
        print(f'GLM: {GLM}, Solver : {Solver_}, dispersion: {dispersion}. N = {N}, J = {J}')
        objs = []
        
        # Indexed by theta and t_pc, contains SP and gamma_max
        pool = {} 
        
    
        
        # theta: number of feture selected 
        for theta in THETA:
            # accuracy percentage requirement
            for t_pc in TAU_percentage: 
                
                # if t_pc in SET[dispersion][GLM]:
                print(f'{GLM}, {dispersion}, {t_pc}')
                # accuracy bound
                tau = tau_min.item() + t_pc*abs(tau_min.item())
                print(f' tau :{tau}, theta: {theta} (+{t_pc*100}% tau_min)')
                
                # Fesible solution 
                SP_feas, gamma_max_feas = feas_sol(theta, t_pc, pool, THETA, TAU_percentage)  if Pareto_Front else ([],0)
                
                #
                # If opt sol available
                #                                                
                if OPT_comparison and dispersion =='dsa':
                    filename = f"D_obj_{GLM}_{dispersion}.xlsx"
                    directory = f'C://Users//{User}//{folder}//D//OPT - Lin//{dispersion}'
                    obj_opt = get_obj_opt_(filename, directory, tau, theta)
                    gap_tol = 1e-5
                    print(f'Optimal reference - ({tau}, {theta}) : {obj_opt}')
                
                #
                # Call and solve the instance
                #
                SP, gamma_max, status = PCD_problem_MAXDISP_(GLM, Solver_,
                                                  dataset,target, 
                                                  features, SQ,P, tau, theta, M, X0,
                                                  dispersion, runs_vns,TimeLimit_vns,
                                                  SP_feas, gamma_max_feas,
                                                  obj_opt= obj_opt, gap_tol=gap_tol)
                
                
                    
                if SP != []:
                    # Save Instance Results: P-models coefficients save
                    save_results_coeff_xlsx('D',Solver_, GLM, dispersion, theta,tau, gamma_max, 
                                            SP,gamma_max, SQ,tau_min,  features, filename = None)
                
                # Record current intance best dispertion found
                objs.append((tau, theta, gamma_max))
                
                # Record all SP and gamma_max associated to theta and t_pc
                
                pool[(theta,t_pc)] = {
                    "gamma_max": gamma_max,
                    "SP": SP
                }
                
                #
                #  Write on log file -- Solver/heristic status
                #
                logfile.write(f"GLM={GLM}, Dispersion={dispersion}, tau={tau:.5f}, theta={theta}, ")
                if Solver_ == 'gurobi_persistent':
                    logfile.write(f"Status: {status[2]}, ObjVal: {status[0]:.5f}, MIPGap: {status[1]:.5f}\n")
                else:
                    logfile.write(f"Status: {status}\n")
                    
           
        #save all instances best dispersion values 
        save_objRes_xlsx('D',Solver_,GLM,dispersion, objs, filename = None)
        
        #close  log file
        logfile.close()
