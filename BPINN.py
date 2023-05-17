"""
This script contains all modules for data handling, normalization and the BPINN model
"""
import numpy as np
import jax.numpy as jnp
from jax import value_and_grad, vmap, grad, pmap
import numpyro
from numpyro.distributions import Gamma, Normal






class BPINN_model:


    def __init__(self, B_sim, M_sim, D_sim, prior_mu, prior_std):
        """Init variables for training data normalization"""
        self.xtr_mean=0
        self.xtr_std=0
        self.ytr_mean=0
        self.ytr_std=0
        self.delta=[]
        self.freq_gen=[]
        self.B_sim=B_sim
        self.M=M_sim
        self.D=D_sim
        self.prior_mu=prior_mu
        self.prior_std=prior_std
        self.wt1=[]
        self.wt2=[]
        self.corr=0
        
        
    
    

    
            
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
            self.freq_gen=y[:,1]
            #self.wt1=y[:,3]
            #self.wt2=y[:,4]
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
       
        
    
        n,m = x.shape
        n=1
    
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
             "h", Normal(self.prior_mu, self.prior_std/jnp.sqrt(prec_nn))#Normal(1.0, 5/jnp.sqrt(prec_nn))
        )  # prior on h
    
        d = numpyro.sample(
             "d", Normal(self.prior_mu, self.prior_std/jnp.sqrt(prec_nn))
        )  # prior on d
        B = numpyro.sample(
             "B", Normal(self.prior_mu, self.prior_std/jnp.sqrt(prec_nn))
        )  # prior on B
        

        
        # precision prior on observations
        prec_obs = numpyro.sample("prec_obs", Gamma(1.0, 0.1))
        
        
            
        
        
      
        with numpyro.plate(
            "data",
            x.shape[0],
            subsample_size=subsample_size,
            dim=-1,
          ):
            def BNN(x):
                return numpyro.deterministic('output', jnp.tanh(x @ w1 + b1) @ w2 + b2)
            def BNN2(x):
                
                u_res=BNN(x)
                temp, u_unn=self.unnormalize(x=x,y=u_res)
                return jnp.squeeze(u_unn)
            
            u_hat=numpyro.sample(
                "y",
                Normal(
                    (BNN(x)),  1.0 / jnp.sqrt(prec_obs)#
                    ),  # 1 hidden layer with tanh activation
                obs=y,
            )
           
            with numpyro.handlers.block():
               u_unn2, u_AD=numpyro.deterministic('AD',vmap(value_and_grad(BNN2))(x))   
            x_un, u_unn=self.unnormalize(x=x, y=u_unn2)
            dP=x_un[:,1]
            
            res_target=np.zeros(len(x_un[:,0]))
            
            self.corr=1/self.xtr_std[0,0]
            def res_calc():
                return numpyro.deterministic('res_calc', self.corr*h*u_AD[:,0]+d*u_unn2+self.B_sim*jnp.sin(self.delta)-dP)
            
            r_hat=numpyro.sample(
                "residual",
                Normal(
                    (res_calc()), 1.0 / jnp.sqrt(prec_obs)
                  ),
                  obs=res_target
                  )

            
            

        
        
        