"""
This script contains all modules for data handling, normalization and the BPINN model
"""
import numpy as np
import jax.numpy as jnp
import numpyro
from numpyro.distributions import Gamma, Normal






class BPINN_model:


    def __init__(self, B_sim):
        """Init variables for training data normalization"""
        self.xtr_mean=0
        self.xtr_std=0
        self.ytr_mean=0
        self.ytr_std=0
        self.delta=[]
        self.B_sim=B_sim
        
        
    
    

    
            
    def normalize(self,x,y, mean=None, std=None):
         """
        Normalize data to zero mean and unit variance

        Parameters
        ----------
        x : jnp array
            Input data of the NN.
        y : jnp array
            Target data of the NN.
        mean : jnp array, 
            Creates the mean. 
        std : jnp array, 
            Creates the standard deviation. 

        Returns
        -------
        jnp array 
            normalized data of x.
        jnp array
            mean of x.
        jnp array
            std of x.
        jnp array
            normalized data of y.
        jnp array
            mean of y.
        jnp array
            std of y.

        """
         if mean is None and std is None:
            self.delta=y[:,0]
            x=jnp.array(x)
            yt=jnp.array(y[:,1])
            xtr_std = jnp.std(x, 0, keepdims=True)
            self.xtr_std = jnp.where(xtr_std == 0, 1.0, xtr_std)
            self.xtr_mean = jnp.mean(x, 0, keepdims=True)
    
            ytr_std = jnp.std(yt, 0, keepdims=True)
            self.ytr_std = jnp.where(ytr_std == 0, 1.0, ytr_std)
            self.ytr_mean = jnp.mean(yt, 0, keepdims=True)
            return (x - self.xtr_mean) / self.xtr_std,self. xtr_mean, self.xtr_std, (yt - self.ytr_mean) / self.ytr_std, self.ytr_mean, self.ytr_std         
    
    
    def unnormalize(self,x,y):
          """
        Reverse the normalization of x,y

        Parameters
        ----------
        x : jnp array
            Input data of NN.
        y : jnp array
            Target data of NN.

        Returns
        -------
        x_unn : jnp array
            unnormalized data of x.
        u_unn : jnp array
            unnormalized data of y.

        """
          x_unn=(x*self.xtr_std)+self.xtr_mean
          u_unn=(y*self.ytr_std)+self.ytr_mean
    
          return x_unn,u_unn
    
    
    
    
    def model(self,x, y=None, hidden_dim=10, subsample_size=100):
        """
        Create the BPINN NN structure

        Parameters
        ----------
        x : jnp array
            Input data of the NN.
        y : jnp array
            Target data of the NN
        hidden_dim : int, optional
            DESCRIPTION. The default is 10.
        subsample_size : int, optional
            DESCRIPTION. The default is 100.

        Returns
        -------
        None.

        """
    
        prec_nn = numpyro.sample(
             "prec_nn", Gamma(1.0, 0.1)
         )  # hyper prior for precision of nn weights, biases and physical priors
        
    
        n, m = x.shape
    
    
        with numpyro.plate("l1_hidden", hidden_dim, dim=-1):
            # prior l1 bias term
            b1 = numpyro.sample(
                 "nn_b1",
                 Normal(
                     0.0,
                     1.0 / jnp.sqrt(prec_nn),
                 ),
             )
            assert b1.shape == (hidden_dim,)
    
            with numpyro.plate("l1_feat", m, dim=-2):
                 w1 = numpyro.sample(
                     "nn_w1", Normal(0.0, 1.0 / jnp.sqrt(prec_nn))
                 )  # prior on l1 weights
                 assert w1.shape == (m, hidden_dim)
    
        with numpyro.plate("l2_hidden", hidden_dim, dim=-1):
             w2 = numpyro.sample(
                 "nn_w2", Normal(0.0, 1.0 / jnp.sqrt(prec_nn))
             )  # prior on output weights
    
        b2 = numpyro.sample(
             "nn_b2", Normal(0.0, 1.0 / jnp.sqrt(prec_nn))
        )  # prior on output bias term
    
    
    
        h = numpyro.sample(
             "h", Normal(1.0, 5/jnp.sqrt(prec_nn))
        )  # prior on h
    
        d = numpyro.sample(
             "d", Normal(1.0, 5/jnp.sqrt(prec_nn))
        )  # prior on d

    
        # precision prior on observations
        prec_obs = numpyro.sample("prec_obs", Gamma(1.0, 0.1))
        with numpyro.plate(
            "data",
            x.shape[0],
            subsample_size=subsample_size,
            dim=-1,
         ):
            
    
            u_hat=numpyro.sample(
                "y",
                Normal(
                     (jnp.tanh(x @ w1 + b1) @ w2 + b2), 1.0 / jnp.sqrt(prec_obs)
                ),  # 1 hidden layer with tanh activation
                obs=y,
            )
    
    
        x_un, u_unn=self.unnormalize(x=x, y=u_hat)
        dP=x_un[:,1]
        dudt=jnp.gradient(u_unn)
        res_target=np.zeros(len(x_un[:,0]))
    
        #calculate the physical part 
        with numpyro.plate(
                "residual_calc",
                x.shape[0],
                subsample_size=subsample_size,
                dim=-1,
    
                ):    
                r_hat=numpyro.sample(
                    "residual",
                    Normal(
                        dudt*h + u_unn*d+self.B_sim*jnp.sin(self.delta)-dP, 1.0 /jnp.sqrt(prec_obs) 
                      ),
                      obs=res_target
                      )
