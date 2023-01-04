# Bayesian Physics-Informed Neural Networks for Robust System Identification of Power System Dynamics

Code related to the submission of
- Simon Stock, Jochen Stiasny, Davood Babazadeh, Christian Becker, Spyros Chatzivasileiadis. "[Bayesian Physics-Informed Neural Networks for Robust System Identification of Power System Dynamics] (https://arxiv.org/abs/2212.11911)." arXiv preprint arXiv:2212.11911(2022).


##  Code structure
The code is structured in the following way:
- `run_sys_id.py` runs the system identification, parameters for the SMIB simulation can be defined
- `SMIB_simulation_loop.py` contains the SMIB Simulation loop 
- `BPINN` contains all data processing and the BPINN model


## Citation

    @misc{stock2022bpinn,
        title={Bayesian Physics-Informed Neural Networks for Robust System Identification of Power System Dynamics},
        author={Simon Stock and Jochen Stiasny and Davood Babazadeh and Christian Becker and Spyros Chatzivasileiadis},
        year={2022},
        eprint={2212.11911},
        archivePrefix={arXiv},
        primaryClass={eess.SY}
    }
 
 ## Related work
 
 The concept of PINNs was introduced by Raissi et al. (https://maziarraissi.github.io/PINNs/) and adapted to power systems by Misyris et al. (https://github.com/gmisy/Physics-Informed-Neural-Networks-for-Power-Systems). The presented code is inspired by these two sources.
 The concept of BPINNs was introduced by Yang et. al "B-PINNs: Bayesian Physics-Informed Neural Networks for Forward and Inverse PDE Problems with Noisy Data", Journal of Computational Physics, 2021, https://doi.org/10.1016/j.jcp.2020.109913 
