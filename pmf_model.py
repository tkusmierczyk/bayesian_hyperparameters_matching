"""Methods calculating E[Y] and E[Y^2] (and therefore also Var[Y]) over prior predictive distribution for Poisson Matrix Factorization."""

import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd
import warnings


def _make_tf(v):
    return tf.Variable(v, dtype=tf.float64) if type(v) is float else v



def empirical_Ey_and_Ey2_np(ct=1.0, rt=1.0, cb=0.1, rb=0.1, 
                         nsamples_latent=100, nsamples_output=3, 
                         N=1, M=1, K=25):        
    """ Returns E_prior[Y] and E_prior[Y^2] for given set of hyperparameters.
        Naive numpy implementation for testing purposes.
    """                  
    ys = []
    for _ in range(nsamples_latent*nsamples_output): 
      theta = np.random.gamma(ct, 1.0/rt, size=K) 
      beta =  np.random.gamma(cb, 1.0/rb, size=K)

      latent = (theta*beta).sum()
      if latent>1e16: 
          warnings.warn("Skipping too large latent value: %s" % latent)
          continue
      y = np.random.poisson(latent)
      ys.append(y)
    expectation = np.mean(ys)
    expectation_squared = np.mean(np.array(ys)**2)
    return expectation, expectation_squared



def empirical_Ey_and_Ey2_tf(ct=1.0, rt=1.0, cb=0.1, rb=0.1, 
                         nsamples_latent=100, nsamples_output=3, 
                         N=1, M=1, K=25):        
    """ Returns E_prior[Y] and E_prior[Y^2] for given set of hyperparameters.
        The outputs are (tf) differentiable w.r.t. hyperparameters. 
    """                  
    if N!=1: warnings.warn("N!=1 will be ignored!")
    if N!=1: warnings.warn("M!=1 will be ignored!")
    ct, rt, cb, rb = _make_tf(ct), _make_tf(rt), _make_tf(cb), _make_tf(rb)
           
    theta = tfd.Gamma(ct, rt).sample((K, nsamples_latent))
    beta =  tfd.Gamma(cb, rb).sample((K, nsamples_latent))
    
    latent = tf.reduce_sum(theta*beta, 0)      
    poisson = tfd.Poisson(rate=latent)
    y_samples = np.random.poisson(latent, size=[nsamples_output, nsamples_latent]) # NO x NL
    
    y_probs = tf.exp( poisson.log_prob(y_samples) )
    total_prob = tf.reduce_sum(y_probs, 0)
    
    conditional_expectation = tf.reduce_sum(y_probs * y_samples, 0) / total_prob
    conditional_expectation_squared = tf.reduce_sum(y_probs * (y_samples**2), 0) / total_prob
    
    expectation = tf.reduce_mean(conditional_expectation)
    expectation_squared = tf.reduce_mean(conditional_expectation_squared)
    
    return expectation, expectation_squared


def empirical_Ey_and_Ey2_tf_reference(ct=1.0, rt=1.0, cb=0.1, rb=0.1, 
                         nsamples_latent=100, nsamples_output=3, 
                         N=1, M=1, K=25):              
    """ Returns E_prior[Y] and E_prior[Y^2] for given set of hyperparameters.
        The outputs are (tf) differentiable w.r.t. hyperparameters. 
        Warning: proof-of-concept code based on loops (very slow);
        for practical purposes use the above version using tensor operations."""                
    ct, rt, cb, rb = _make_tf(ct), _make_tf(rt), _make_tf(cb), _make_tf(rb)

    expectation = 0.0
    expectation_squared = 0.0
    for n in range(nsamples_latent):            
        # sample latent variables
        theta = tfd.Gamma(ct, rt).sample((N,K))
        beta =  tfd.Gamma(cb, rb).sample((K,M))
        latent = tf.matmul(theta, beta) 

        # for each latent sample y-s
        total_prob = 0.0
        conditional_expectation = 0.0
        conditional_expectation_squared = 0.0
        latent = latent[0,0] # take just one cell (comment it out to take a mean over all cells)
        poisson = tfd.Poisson(rate=latent)

        #y_samples = list(i for i in range(nsamples_output))
        y_samples = np.array([np.random.poisson(latent) for _ in range(nsamples_output)])
        y_probs = [tf.exp(poisson.log_prob(y)) for y in y_samples]
        #assert_valid(np.array(y_probs))
        for m in range(nsamples_output):
            y, prob = y_samples[m], y_probs[m]   
            total_prob += prob                
            conditional_expectation += prob * y
            conditional_expectation_squared += prob * (y*y)
        expectation += conditional_expectation/total_prob
        expectation_squared += conditional_expectation_squared/total_prob      
          
    expectation = expectation/(nsamples_latent)
    expectation_squared = expectation_squared/(nsamples_latent)    
    return expectation, expectation_squared


def create_moments_estimator(K=25, ESTIMATOR_NO=-1, N=1, M=1, 
                             empirical_Ey_and_Ey2 = empirical_Ey_and_Ey2_tf):
    

    def theoretical_moments(ct, rt, cb, rb):
        mt, st = ct/rt, np.sqrt(ct) / rt
        mb, sb = cb/rb, np.sqrt(cb) / rb
        e, var = K * mt*mb, K * (mt*mb + (mb*st)**2 + (mt*sb)**2 + (st*sb)**2)        
        return e, var
      

    def empirical_moments_EV(a, b, c, d, 
                             nsamples_latent, nsamples_output, 
                             N=N, M=M, K=K): 
        """ Returns empirical estimates of expectation & variance.""" 
        expectation, expectation_squared = empirical_Ey_and_Ey2(a, b, c, d, 
                                                    nsamples_latent=nsamples_latent, 
                                                    nsamples_output=nsamples_output, 
                                                    N=N, M=M, K=K)
        variance = expectation_squared - expectation**2        
        expectation, variance = tf.reduce_mean(expectation), tf.reduce_mean(variance) 
        return expectation, variance


    def empirical_moments_EV_decoupled(a, b, c, d, 
                                       nsamples_latent, nsamples_output, 
                                       N=N, M=M, K=K):  
        """ Returns empirical estimates of expectation & variance.
            Decoupled E & V but E[y] and E[y^2] for V computed together.
        """
        expectation, _ = empirical_moments_EV(a, b, c, d, 
                                              nsamples_latent=nsamples_latent, 
                                              nsamples_output=nsamples_output, 
                                              N=N, M=M, K=K)
        _, variance = empirical_moments_EV(a, b, c, d, 
                                           nsamples_latent=nsamples_latent, 
                                           nsamples_output=nsamples_output, 
                                           N=N, M=M, K=K)
        return expectation, variance
      
        
    def empirical_moments_V_decoupled(a, b, c, d, 
                                      nsamples_latent, nsamples_output, 
                                      N=N, M=M, K=K):  
        """ Returns empirical estimates of expectation & variance.
            Uses the same E[y] for both E and V; E[y^2] independent of E[y]. 
        """
        expectation, _ = empirical_Ey_and_Ey2(a, b, c, d, 
                                              nsamples_latent=nsamples_latent, 
                                              nsamples_output=nsamples_output, 
                                              N=N, M=M, K=K)
        _, expectation_squared = empirical_Ey_and_Ey2(a, b, c, d, 
                                                      nsamples_latent=nsamples_latent, 
                                                      nsamples_output=nsamples_output, 
                                                      N=N, M=M, K=K)    
        variance = expectation_squared - expectation**2        
        expectation, variance = tf.reduce_mean(expectation), tf.reduce_mean(variance)    
        
        return expectation, variance    


    def empirical_moments_fully_decoupled(a, b, c, d, 
                                          nsamples_latent, nsamples_output, 
                                          N=N, M=M, K=K):  
        """ Returns empirical estimates of expectation & variance.
            Computes E[y] for E independent of E[y] for V. E[y^2] independent of E[y]. 
        """        
        expectation, _ = empirical_Ey_and_Ey2(a, b, c, d, 
                                              nsamples_latent=nsamples_latent, 
                                              nsamples_output=nsamples_output, 
                                              N=N, M=M, K=K)
        _, expectation_squared = empirical_Ey_and_Ey2(a, b, c, d, 
                                                      nsamples_latent=nsamples_latent, 
                                                      nsamples_output=nsamples_output, 
                                                      N=N, M=M, K=K)    
        variance = expectation_squared - expectation**2        
        variance = tf.reduce_mean(variance) # empirical
        
        expectation, _ = empirical_Ey_and_Ey2(a, b, c, d, 
                                              nsamples_latent=nsamples_latent, 
                                              nsamples_output=nsamples_output, 
                                              N=N, M=M, K=K)
        expectation = tf.reduce_mean(expectation)
        
        return expectation, variance   


    NO2ESTIMATOR = {-1: theoretical_moments,
                     0: empirical_moments_fully_decoupled, 
                     1: empirical_moments_EV_decoupled, 
                     2: empirical_moments_EV, 
                     3: empirical_moments_V_decoupled}
    if ESTIMATOR_NO not in NO2ESTIMATOR: 
        raise Exception("Wrong estimator number! Try: %s" % NO2ESTIMATOR)
    estimator = NO2ESTIMATOR[ESTIMATOR_NO]
    return estimator


empirical_Ey_and_Ey2 = empirical_Ey_and_Ey2_tf

