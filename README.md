# Source Code of Experiments in "On Disperse Generalized Linear Models"

This repository contains the source code used for the experiments in the paper:

> **"On Disperse Generalized Linear Models"**  
> by *Emilio Carrizosa, Renato De Leone, and Marica Magagnini*

---

The codes in this repository implement the (A-PDP-GLM) and (D-PDP-GLM) problems, as well as the heuristic algorithm proposed in the paper.  
It includes all datasets and the supporting functions required to reproduce the experiments described in *On Disperse Generalized Linear Models*.

---

## Repository Structure

### Folders
- **Datasets/**  
  Contains the raw datasets used for the experiments.

- **P_CD_problem_MaxAcc/**  
  Contains the `.py` files implementing and solving (both with heuristic and optimal solvers) the (A-PDP-GLM) problem. 

- **P_CD_problem_MaxDisp/**  
  Contains the `.py` files implementing and solving (both with heuristic and optimal solvers) the (D-PDP-GLM) problem.

- **SQ/**  
  Contains the reference models (`.csv` files) used in the experiments, and the generator `.py` file.

### Other Files
- **BostonHousing.py**, **SeoulBike.py** → Preliminary data handling scripts.  
- **Funs.py**, **f_print.py** → Supporting functions for the experiments.  
- **Runner_MaxDisp.py** → Main file to compute all dispersion parameters in the experiments, i.e., sequentially solve instances of the (D-PDP-GLM) problem.  
- **Runner_MaxAcc.py** → Main file to obtain all results in the experiments, i.e., sequentially solve instances of the (A-PDP-GLM) problem.
