"""
This script runs the SMIB simulation to create data that can be used for estimation
"""

import numpy as np
from scipy import integrate



class SmibSystem:

    def __init__(self, m, d, B, P):
        """Init variables for SMIB simulation"""
        self.m = m
        self.d = d
        self.B = B
        self.P = P

    def update_function(self, time, state):
        """Update the function by one step"""

        delta, omega = state

        d_dt_delta = omega
        d_dt_omega = 1 / self.m * (self.P - self.d * omega - self.B * np.sin(delta))
        return np.array([d_dt_delta, d_dt_omega])

    def simulate_trajectory(self, state_initial, time_end, time_step_size, K):
        """simulate the full trajectory length and prepare the results"""
        time_start = 0.0
        n_time_steps = int(np.round(time_end / time_step_size, decimals=6)) + 1
        evaluation_times = np.linspace(0, time_end, n_time_steps)

        assert len(state_initial.flatten()) == 2, 'The state should have two elements, delta and omega.'

        ode_solution = integrate.solve_ivp(self.update_function,
                                           t_span=[time_start, time_end],
                                           y0=state_initial.flatten(),
                                           t_eval=evaluation_times,
                                           rtol=1e-13)

        assert ode_solution.success

        time_results = ode_solution.t.reshape((-1, 1))
        time_results=np.squeeze(time_results)
        state_results = ode_solution.y.T
        
        #Add the noise
        state_results= np.transpose(np.array([state_results[:,0]+np.mean(abs(state_results[:,0]))*K*np.random.standard_normal(len(state_results)),
                                              state_results[:,1]+np.mean(abs(state_results[:,1]))*K*np.random.standard_normal(len(state_results))]))
        power_results = np.array([[self.P]]).repeat(repeats=time_results.shape[0], axis=0)
        power_results= power_results[:,0]+np.transpose(K*(power_results[:,0])*np.random.standard_normal(len(power_results)))
        return time_results, state_results, power_results
   

def main(M_INIT, D_INIT, B, TIME_STEP_SIZE, K, nn_in, nn_out, TRAJECTORY_LENGTH):
    """
    Run the SMIB simulation

    Parameters
    ----------
    M_INIT : double
        inertia parameter parameter.
    D_INIT : double
        damping parameter.
    B : double
        susceptance matrix entry.
    TIME_STEP_SIZE : double
        step width of the simulation.
    K : double
        Noise level.
    nn_in : np array
        input data for the NN.
    nn_out : np array
        target data for the NN.
    TRAJECTORY_LENGTH : double
        length of the simulation.

    Returns
    -------
    nn_in : np array
        Input for the NN.
    nn_out : np array
        Target for the NN.

    """
    test_system = SmibSystem(m=M_INIT, d=D_INIT, B=B, P=0.1)
    time_end = TRAJECTORY_LENGTH-TIME_STEP_SIZE
    state_initial = np.array([0.0, 0.0])
    time, state, power = test_system.simulate_trajectory(
        state_initial=state_initial,
        time_end=time_end,
        time_step_size=TIME_STEP_SIZE,
        K=K
        )


  
    DP=M_INIT*np.gradient(state[:,1])+D_INIT*state[:,1]+B*np.sin(state[:,0])
    nn_in=np.array([time, DP])
    nn_in=np.transpose(np.squeeze(nn_in))
    nn_out=state
    return nn_in, nn_out
   