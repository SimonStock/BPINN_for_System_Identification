"""
This script runs the system identification with BPINNs
"""
import argparse
from functools import partial
import numpy as np
from jax import random
import numpyro
from numpyro.contrib.einstein import RBFKernel, SteinVI
from numpyro.infer import  Trace_ELBO, init_to_median
from numpyro.infer.autoguide import AutoDelta
from numpyro.optim import Adagrad


import smib_simulation_loop as sim
from BPINN import BPINN_model







#%% Define parameters for the SMIB simulation

# Swing eqaution parameterss
M_INIT=0.3
D_INIT=0.15
B=0.2

# Parameters under test (Noise ~ K)
K=0.0
TRAJECTORY_LENGTH=27

TIME_STEP_SIZE=0.1

#%% Main module containing data handling, definition of BPINN model and training
def main(args):

    nn_in=[]
    nn_out=[]
    
    # Perform SMIB system simulation
    nn_in, nn_out= sim.main(M_INIT, D_INIT, B, TIME_STEP_SIZE, K, nn_in, nn_out, TRAJECTORY_LENGTH)
    
    model=BPINN_model(B_sim=B)
    inf_key, pred_key, data_key = random.split(random.PRNGKey(args.rng_key), 3)
    
    # Normalize data to zero mean unit variance!
    x, xtr_mean, xtr_std, y, ytr_mean, ytr_std = model.normalize(x=nn_in, y=nn_out)
    args.subsample_size=len(x[:,0])
    rng_key, inf_key = random.split(inf_key)
        
    
    # Define the BPINN model
    stein = SteinVI(
            model.model,
            AutoDelta(model.model, init_loc_fn=partial(init_to_median)),
            Adagrad(0.05),
            Trace_ELBO(20),  
            RBFKernel(),
            repulsion_temperature=args.repulsion,
            num_particles=args.num_particles,
            )


           
    # Run the training
    result = stein.run(
             rng_key,
             args.max_iter,
             x,
             y,
             hidden_dim=args.hidden_dim,
             subsample_size=args.subsample_size,
             progress_bar=args.progress_bar,
             )



    # Extract the mean and standard deviation of m, d estimates     
    net_param=result[0]
    m_mean=np.mean(np.array(net_param['h_auto_loc']))
    print('m_mean:', m_mean)
    m_std=np.std(np.array(net_param['h_auto_loc']))
    print('m_std:', m_std)
    d_mean=np.mean(np.array(net_param['d_auto_loc']))
    print('d_mean:', d_mean)
    d_std=np.std(np.array(net_param['d_auto_loc']))
    print('d_std:', d_std)

       
if __name__ == "__main__":
    from jax.config import config

    config.update("jax_debug_nans", True)        
    parser = argparse.ArgumentParser()
    parser.add_argument("--subsample-size", type=int, default=270)
    parser.add_argument("--max-iter", type=int, default=5000)
    parser.add_argument("--repulsion", type=float, default=1.0)
    parser.add_argument("--verbose", type=bool, default=True)
    parser.add_argument("--num-particles", type=int, default=90)
    parser.add_argument("--progress-bar", type=bool, default=True)
    parser.add_argument("--rng-key", type=int, default=142)
    parser.add_argument("--device", default="cpu", choices=["gpu", "cpu"])
    parser.add_argument("--hidden-dim", default=10, type=int)
        
    args = parser.parse_args()

    numpyro.set_platform(args.device)

    main(args)