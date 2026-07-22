
"""
@author: Marica Magagnini

This file contains the main function to perform the experiments of the project
"On disperse GLM."

Most function can be employed in both of the two cases:
    - maximizing the accuracy. 
    - maximazin the dispersion.
    
Case-specific function has an appropriate name. 
"""

import pandas as pd
import os
from pyomo import environ as pym
from f_print import get_obj_sol_
import gurobipy as GRB



#
#  Dataset
# 
def Dataset_selection(GLM,name = None):
    """
    This function select the dataset for the experiments.
    Note: complete path to run
    """
    path = f'C:/Users/.../Datasets/'
    task = None
    if GLM == 'Lin':
        if name =='ITT':
            from Thermo import data          
        elif name == 'CC':
            from CommCrime import data          
        else:
            from BostonHousing import data
                   
   
    elif GLM == 'Log':
        from BostonHousing import data
        task = 'classification'
    elif GLM == 'Poi':
        from SeoulBike import data
       
    return data(path,task)

#
# Known models
#
def S_Q_(GLM, features, name = None):
    """
    
    Parameters
    ----------
    GLM : string
        Generalized Linear Model: 'Lin', 'Log', 'Poi'
    features : array of strings
        Features names of the dateset corresponding to the selectes GLM.

    Returns
    -------
    SQ : DataFrame of Q rows 
        Each row represents a GLM already known. The columns are the coefficients 
        of the Q known models.
    SQ_obj : Series
        It contains the accuracy of the Q models.

    Note : complete path to run
    """
    if name and GLM =='Lin':
        full_path =f'C:/Users/.../SQ/SQ_{GLM}_{name}.xlsx'
    else:
        full_path =f'C:/Users/.../SQ/SQ_{GLM}.xlsx'
    
    df = pd.read_excel(full_path)
    
    SQ_obj = df['tau_min']
    subset = ['bias'] + list(features)
    SQ = df[subset]
    
    return SQ, SQ_obj


###############################################################################
# Maximize dispersion : call and solve a specific instance
###############################################################################
    
def PCD_problem_MAXDISP_(GLM, Solver_,
                 dataset,target, 
                 features, SQ,P, tau, theta, M, X0,
                 dispersion, runs_vns,TimeLimit_vns,
                 SP_feas, obj_feas,
                 obj_opt=None, gap_tol=None):
    """
    Parameters
    ----------
    GLM : string
        Generalized Linear Model: 'Lin', 'Log', 'Poi'
    Solver_ : string
        "gurobi_persistent" (option available only for 'Lin') or "ipopt" 
    dataset : dataframe
        Only feature values.
    target : Series
        Target values of the dataset.
    features : array of strings
        FEtures names.
    SQ : DataFrame of Q rows 
        Each row represents a GLM already known. The columns are the coefficients 
        of the Q known models.
    P : int
        Number of models to be built.
    tau : float
        accuracy threshold.
    theta : int
        Number non-zero features in the new P-models.
    M : int
        Big-M parameter.
    X0 : Series or dataframe
        Selected instance of the dataset used in case of 'o1/o2' dispersion. 
    dispersion : string
        'l1', 'l2', 'dsa', 'o1'
    runs_vns : int
        Heuristic multi-start parameter: Number of times to restart the VNS 
    TimeLimit_vns : int
        VNS time-limit for a sing run. 

    Returns
    -------
    SP : disct of dict
        It containts P-dicts, where each one is a new model. Each dict contains 
        the model coefficients (bias included). 
    SP_obj : float
        Maximal dispepersion computed for the set of P-models.
    status : String or NoneType
        It containts the gurobi final status when it is used as solver. 

    """
    
    N,J = dataset.shape     # N: number of dataset elements, number of features
    J1 = J+1                # number of fetures + bias term
    status = None
    SP, SP_obj = (SP_feas, obj_feas)
    type_sol = ' (saved last best feas sol) '  
    ###########################################################################
    # Linear Regression - Exact computation
    ###########################################################################
    if GLM == 'Lin' and Solver_ == 'gurobi_persistent':       
        
        #
        # Instance call
        #
        from P_CD_problem_MaxDisp.PCD_MAX_dispersion_Lin import P_max_dispersion_Lin_
        instance = P_max_dispersion_Lin_(dispersion, dataset, target, features, M, 
                         SQ, P, tau, theta, X0)
        
        # 
        # Solver
        #
        solver = pym.SolverFactory(Solver_)
        solver.set_instance(instance)
        
        #Solver parameters
        solver.options['TimeLimit'] =1800 
       
        #
        # Results
        #
        solver.solve(tee = True)
        
        #
        # Solver status
        #
        grb_model = solver._solver_model

        status_code = grb_model.Status
        sol_count = grb_model.SolCount if hasattr(grb_model, "SolCount") else 0
        
        objval = float("nan")
        mipgap = float("nan")
        # Obj value and MIPGap, if available
        if sol_count > 0 :
            objval = grb_model.ObjVal
            mipgap = grb_model.MIPGap if hasattr(grb_model, "MIPGap") else float('nan')
            
            global get_obj_sol_
            SP_obj_gurobi, SP_gurobi = get_obj_sol_(instance)
            
            
            if  SP_obj_gurobi > SP_obj:
                SP, SP_obj = (SP_gurobi, SP_obj_gurobi)
                type_sol = ' (Gurobi save) '
                
                
        # Interpretation of the status code
        status_msg = {
            GRB.GRB.OPTIMAL: "Optimal solution found",
            GRB.GRB.TIME_LIMIT: "Time limit reached",
            GRB.GRB.INFEASIBLE: "Infeasible model",
            GRB.GRB.UNBOUNDED: "Unbounded model",
            GRB.GRB.INF_OR_UNBD: "Infeasible or unbounded",
            GRB.GRB.INTERRUPTED: "Manually interrupted",
            }.get(status_code, f"Status unknown (code={status_code})")
        
        status_msg = status_msg + type_sol
        status = (objval,mipgap,status_msg)
        

    ###########################################################################
    # Linear, Logistic,  Poisson Regression - Heuristic Results
    ###########################################################################
    else:
        # Select local solver maximum number of iteration 
        solver_Iterations = 2000
        
        # Call heuristic strategy
        from P_CD_problem_MaxDisp.VNS_MaxDisp import VNS 
        
        
        #
        # VNS parameters
        #
        K = J1 #max ~ PxJ1


        
        status = 'FAIL : Heuristic did NOT find a solution better than the feasible initial one.'
        SP_objtime_ev = pd.DataFrame([{'time': 0,'obj': obj_feas}])
        
        for run in range(runs_vns):
            

            RES, FSPvns, betaPvns, obj_vns, obj_time_ev_vns = VNS(GLM, Solver_,
                                                                 K,TimeLimit_vns,
                                                                 dataset, target, features, SQ, 
                                                                 P,J1,theta, tau,dispersion, 
                                                                 X0, solver_Iterations,
                                                                 obj_feas, obj_opt, gap_tol)
                        
            
            # Select the best result among all the VNS runs (largest dispersion value)
            if obj_vns > SP_obj: 
                SP_obj = obj_vns         
                SP = betaPvns    
                SP_objtime_ev = pd.DataFrame(obj_time_ev_vns)
                status = 'SUCCESS : Heuristic FOUND a solution better than the feasible initial one.'
        
        
                
        #Save obj evolution over time
        save_dir = os.path.join('.', f'D_HEUR_{GLM}_{dispersion}')
        os.makedirs(save_dir, exist_ok=True)
        filename = f"D_objtime_HEUR_{GLM}_{dispersion}_tau{tau}_theta{theta}.xlsx"
        filepath = os.path.join(save_dir, filename)
        SP_objtime_ev.to_excel(filepath, index=False)
            
    return SP, SP_obj, status

###############################################################################
# Maximize accuracy : call and solve a specific instance
###############################################################################


# Parameters for Max Acc case
def params_(GLM, dispersion, directory = None):
    """
    Function to read the parameters (tau, theta, gamma_max) from a file.
    Parameters
    ----------
    GLM : string
        Generalized Linear Model: 'Lin', 'Log', 'Poi'
    dispersion : string
        'l1', 'l2', 'dsa', 'o1'

    Returns
    -------
    parameters : dataframe
        Columns: tau (accuracy), theta (number feature selected), gamma_max 
                (maximal dipersion computed for the couple (tau,theta)).

    """
    
    # Build directory if not provided
    
    if not directory :
        directory = f'C:/Users/../params/'
    nome_file = f"D_Obj_{GLM}_{dispersion}.xlsx"
    percorso_completo = os.path.join(directory, nome_file)

    # Verify if file exists
    if not os.path.exists(percorso_completo):
        print(f"File not found: {percorso_completo}")
        return

    # Read the file based on its extension
    parameters = pd.read_excel(percorso_completo)

    
    return parameters


def PCD_problem_MAXACC_(GLM, Solver_,
                 dataset,target,
                 features, SQ, P, gamma, tau, theta, M, X0,
                 dispersion, runs_vns,TimeLimit_vns,
                 SP_feas, obj_feas, 
                 obj_opt=None, gap_tol=None):
    """

    Parameters
    ----------
    GLM : string
        Generalized Linear Model: 'Lin', 'Log', 'Poi'
    Solver_ : string
        "gurobi_persistent" (option available only for 'Lin') or "ipopt" 
    dataset : dataframe
        Only feature values.
    target : Series
        Target values of the dataset.
    features : array of strings
        FEtures names.
    SQ : DataFrame of Q rows 
        Each row represents a GLM already known. The columns are the coefficients 
        of the Q known models.
    P : int
        Number of models to be built.
    gamma : float (or int in case of dispersion = 'dsa')
        dispersion threshold.
    tau : float
        accuracy threshold.
    theta : int
        Number non-zero features in the new P-models.
    M : int
        Big-M parameter.
    X0 : Series or dataframe
        Selected instance of the dataset used in case of 'o1/o2' dispersion. 
    dispersion : string
        'l1', 'l2', 'dsa', 'o1'
    runs_vns : int
        Heuristic multi-start parameter: Number of times to restart the VNS 
    TimeLimit_vns : int
        VNS time-limit for a sing run. 


    Returns
    -------
    SP : dict of dict
        It containts P-dicts, where each one is a new model. Each dict contains 
        the model coefficients (bias included). 
    SP_obj : float
        Maximal accuracy (min error/los) computed for the set of P-models.
    status : String or NoneType
        It containts the gurobi final status when it is used as solver.     

    """
    
    
    N,J = dataset.shape
    J1 = J+1
    status = None
    SP, SP_obj = (SP_feas, obj_feas)
    ###########################################################################
    # Linear Regression - Exact computation
    ###########################################################################
    if GLM == 'Lin' and Solver_ == 'gurobi_persistent':       
        
        #
        # Instance call
        #
        from P_CD_problem_MaxAcc.PCD_Max_Acc_Lin import P_cond_disp_
        instance = P_cond_disp_(dispersion, dataset, target, features, 
                                  SQ, P, gamma, tau, theta, M, X0)
        # 
        # Solver
        #
        solver = pym.SolverFactory(Solver_)
        
        solver.set_instance(instance)
               
        #Solver parameters
        solver.options['TimeLimit'] =1800
       
        #
        # Results
        #
        solver.solve(tee = True)
        
        #
        # Solver status
        #
        
        grb_model = solver._solver_model

        
        status_code = grb_model.Status
        sol_count = grb_model.SolCount if hasattr(grb_model, "SolCount") else 0

        objval = float("nan")
        mipgap = float("nan")
        # Obj value and MIPGap, if available
        if sol_count > 0:
            objval = grb_model.ObjVal
            mipgap = grb_model.MIPGap if hasattr(grb_model, "MIPGap") else float('nan')
            
            global get_obj_sol_
            SP_obj_gurobi, SP_gurobi = get_obj_sol_(instance)
            
            
            if  SP_obj_gurobi < SP_obj:
                SP, SP_obj = SP_gurobi, SP_obj_gurobi
                
        # Interpretation of the status code
        status_msg = {
            GRB.GRB.OPTIMAL: "Optimal solution found",
            GRB.GRB.TIME_LIMIT: "Time limit reached",
            GRB.GRB.INFEASIBLE: "Model is infeasible",
            GRB.GRB.UNBOUNDED: "Model is unbounded",
            GRB.GRB.INF_OR_UNBD: "Infeasible or unbounded",
            GRB.GRB.INTERRUPTED: "Interrupted manually",
            }.get(status_code, f"Unknown status (code={status_code})")
                
        status = (objval,mipgap,status_msg)
        

    ###########################################################################
    # Linear, Logistic,  Poisson Regression - Heuristic Results
    ###########################################################################
    else:
        # Select local solver maximum number of iteration
        solver_Iterations = 1000
        
        from P_CD_problem_MaxAcc.VNS_Max_Acc import VNS

        #
        # VNS parameters
        #
        K =  J1 #max ~ PxJ1
        
        
        status = 'FAIL : Heuristic did NOT find a solution better than the feasible initial one.'
        SP_objtime_ev = pd.DataFrame([{'time': 0,'obj': obj_feas}])
        
        for run in range(runs_vns):
            
            RES, FSPvns, betaPvns, obj_vns, obj_time_ev_vns = VNS(GLM, Solver_, 
                                                                 K,TimeLimit_vns,  
                                                                 dataset, target, features, SQ,
                                                                 P,J1,theta, gamma, tau,dispersion, 
                                                                 X0, solver_Iterations,
                                                                 obj_feas, obj_opt, gap_tol)
                                
            # SP_obj nel caso euristico è il migliore ottenuto sui run
            if obj_vns < SP_obj: 
                SP_obj = obj_vns         
                SP = betaPvns               # SP of of all the runs
                SP_objtime_ev = pd.DataFrame(obj_time_ev_vns)
                status = 'SUCCESS : Heuristic FOUND a solution better than the feasible initial one.'
        
        
        

        #Save obj evolution over time
        save_dir = os.path.join('.', f'A_HEUR_{GLM}_{dispersion}')
        os.makedirs(save_dir, exist_ok=True)
        filename = f"A_objtime_HEUR_{GLM}_{dispersion}_tau{tau}_theta{theta}_gamma{gamma}.xlsx"
        filepath = os.path.join(save_dir, filename)
        SP_objtime_ev.to_excel(filepath, index=False)
        
    return SP, SP_obj, status


###############################################################################
# Save Results
###############################################################################

#Coefficients
def save_results_coeff_xlsx(AD, Solver_, GLM, dispersion, theta, tau, gamma, 
                            SP,SP_obj, SQ, SQ_obj,  features, filename = None):
    """
    

    Parameters
    ----------
    AD : string
        'D' - maximal dispersion 
        'A' - maximal accuracy
    Solver_ : string
        "gurobi_persistent" (option available only for 'Lin') or "ipopt". 
        It defines the type of solution -- > 'gurobi_persistent': 'OPT', 'ipopt': 'HEUR'
    GLM : string
        Generalized Linear Model: 'Lin', 'Log', 'Poi'
    dispersion : string
        'l1', 'l2', 'dsa', 'o1'
    tau : float
        accuracy threshold.
    theta : int
        Number non-zero features in the new P-models.
    gamma : float
        Dipersion lower bound (only when AD = 'A')
    SP : disct of dict
        It containts P-dicts, where each one is a new model. Each dict contains 
        the model coefficients (bias included). 
    SP_obj : float
        If AD = 'D' :  Maximal dispepersion computed for the set of P-models.
        If AD = 'A' :  Maximal accuracy (sum (or mean)) computed for the set of P-models.
    SQ : DataFrame of Q rows 
        Each row represents a GLM already known. The columns are the coefficients 
        of the Q known models.
    SQ_obj : Series
        It contains the accuracy of the Q models.
    features : array of strings
        Features names of the dateset corresponding to the selectes GLM.
    filename : TYPE, optional
        Name of the file to create. The default is None.
        

    Returns
    -------
    Save results in a .xlsx file. 
    The first set of P rows containts the results of the set of models 
    built for a spacific combination of parameters. 
    The second set of Q models are the already known models.
    File name framework:
        filename = f"{AD}_Coeff_{Type}_{GLM}_{dispersion}_tau{tau}_theta{theta}_gamma{gamma}.xlsx"
    
    Directory: {AD}_{Type}_{GLM}_{dispersion}
    
    The 'Obj' column is :
        - the maximal dispersion (or gamma_max) when AD = 'D';
        - the maximal accuracy when AD = 'A'.

    """
    #
    # Build Directory 
    #
    Type = {'gurobi_persistent': 'OPT', 'ipopt': 'HEUR'}.get(Solver_, None)
    save_dir = os.path.join('.', f'{AD}_{Type}_{GLM}_{dispersion}')
    os.makedirs(save_dir, exist_ok=True)
    
    #
    # Filename if not provided
    #
    if filename is None:
        filename = f"{AD}_Coeff_{Type}_{GLM}_{dispersion}_tau{tau}_theta{theta}_gamma{gamma}.xlsx"
       
    full_path = os.path.join(save_dir, filename)

    
  
    dati = []

   
    # Append SP models with the complete obj function (sum of the P)        
    P = len(SP)
    for p in range(P):
        row = [p+1]+ [SP[p][f] for f in   ['bias'] + list(features)] + [SP_obj]
        dati.append(row)
    
    # Append SQ models
    Q = SQ.shape[0]
    for q in range(Q):
        row = [p+2] + [SQ.iloc[q][f].item() for f in   ['bias'] + list(features)] +[SQ_obj[q]]
        dati.append(row)
    

    # Nomi colonne: index, GML, beta_0 ... beta_J1, Train_score, Test_score
    colonne = [f'{GLM}_P1:{P}_Q{P+1}:']+ ['bias'] + list(features)  + ['Obj']
    
    
    # Save dataframe to .xlsx
    df = pd.DataFrame(dati, columns=colonne)
    df.to_excel(full_path, index=False)

    print(f"File coeff salvato come '{filename}' in {full_path}")
    
    
#Objective values   
def save_objRes_xlsx(AD, Solver_, GLM,dispersion, parms_objs, filename = None):
    """

    Parameters
    ----------
    AD : string
        'D' - maximal dispersion 
        'A' - maximal accuracy
    Solver_ : string
        "gurobi_persistent" (option available only for 'Lin') or "ipopt". 
        It defines the type of solution -- > 'gurobi_persistent': 'OPT', 'ipopt': 'HEUR'
    GLM : string
        Generalized Linear Model: 'Lin', 'Log', 'Poi'
    dispersion : string
        'l1', 'l2', 'dsa', 'o1'
    parms_objs : tuple 
        If AD = 'D' : (tau, theta, obj) where obj is the maximal dispersion
        If AD = 'A' : (tau, theta, gamma, obj) where obj is the maximal accuracy
    filename : TYPE, optional
        Name of the file to create. The default is None.

    Returns
    -------
    Save results in a .xlsx file. 
    Directory: {AD}_{Type}_{GLM}_{dispersion}
    File name framework:
        filename = f"{AD}_Obj_{GLM}_{dispersion}.xlsx"
        
    Each row represnts a different experiments where the parameters are record in the 
    firsts columns. Last column is the value of the objective function. 
        

    """
  
    #
    # Build Directory 
    #
    Type = {'gurobi_persistent': 'OPT', 'ipopt': 'HEUR'}.get(Solver_, None)
    save_dir = os.path.join('.', f'{AD}_{Type}_{GLM}_{dispersion}')
    os.makedirs(save_dir, exist_ok=True)
    
    
    #
    # Filename if not provided
    #
    if filename is None:
        filename = f"{AD}_Obj_{GLM}_{dispersion}.xlsx"
    
    full_path = os.path.join(save_dir, filename)

    
    # Build Dataframe
    if AD == 'A':
        df = pd.DataFrame(parms_objs, columns=['tau','theta', 'gamma', 'SP_obj'])
    if AD == 'D':
        df = pd.DataFrame(parms_objs, columns=['tau','theta', 'gamma_max'])
   
        
    # Save dataframe to .xlsx
    df.to_excel(full_path, index=False)
    print(f"File objs salvato come '{filename}'")

   
   