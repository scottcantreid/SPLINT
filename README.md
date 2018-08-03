# T-Dependent-NEF
Temperature robustness can be achieved in mixed analog/digital computers in the neuromorphic NEF framework by changing neural weights as a function of temperature.

How to use this package:

## (I) Simulating neuron tuning curves:
Use gen_neurons(Nraw, Q, sig, R, del_bad=False) to generate temperature dependent tuning curves of Nraw neurons over a domain of Q input  at R different temperatures. del_bad=True will lead to a slightly smaller neuron yield, as "bad" neurons which rarely fire will be cut out of the neural population. One can compare results between del_bad=True and del_bad=False to see the effect of "bad" neurons. Preliminary results suggest that "bad" neurons are actually quite valuable for temperature robustness and decode accuracy.

```
Tarr, A, iin = tempy.gen_neurons(Nraw = 5000, Q = 1000, sig = 0.1, R = 30)
```

## (II) The quick and easy way of solving for optimal decode vectors for function f:
#### (II-A) Least Squares decode vector centered at temperature Tarr[t]
```  
d_ls = tp.ls_dstar(A, f, sigma, t = t)
```
#### (II-B) LSAT decode vector
``` 
d_lsat = (A, f, sigma, Tarr)
```          
If you want to quickly compute decode vectors for many different functions, it is recommended that you use the transformation matrix method, which is explained later.
        
####  (II-C) Lint Decode Vector
```
d_lint = lint(A, f, sigma, Tarr).
```
Again, the transformation matrix method is handy for computing decode vectors for many functions.
#### (II-D) Splint Decode Vector
```
d_splint = splint(A, f, sigma, Tarr, k)
```
where k is the number of neurons which will have linear-in-temperature weights. The default is to sort neurons by the magnitude of their lint d_0 weights. If you want to use a custom neuron sort, input it as parameter "sorted_neurons." For more control, it is highly recommended to use the transformation matrix method.
        
#### (II-E) Quint 
```
d_quint = quint(A, f, sigma, Tarr)
```
        
#### (II-F) Squint
```
d_squint = squint(A, f, sigma, Tarr, k)
```
Where k is the number of neurons with temperature-dependent weights. This works, however, it is recommended to use the transformtion matrix method, as the optimal way to choose a sparse population for quint weights is still an open question.

## (III) Computing Error
We now know how to compute decode coefficients *D* for a function *f*. How do we compute the error between our decoded function and the target function?

```
error = error_t(d_coeffs, A, f, Tarr)
```
where d_coeffs is a stacked vector of decode coefficients (i.e. [d0, d1, d2] for a quint solution).
*error_t* will return an array of error values for each temperature in Tarr.
        
## (IV) Transformation matrix method for non-sparse solutions
LSAT, Lint, Quint, and polynomial-order Pint weights can all be unified in a simple mathematical framework which characterizes a linear transformation from R^Q, the discretized function space, to R^((P+1)N), the space of P^th polynomial order decode weight coefficients.
    
A temperature-dependent decode vector may be expressed as a polynomial series summation:
```
d(T) = d_0 + T d_1 + T^2 d_2 + ... + T^P d_P
```
Each coefficient d_i is a vector of length N, where N is the number of neurons in the population. We can create a stacked vector 
```
d_coeffs = [d_0, d_1, ..., d_P]
```    
Defined in this way, the solution to the P^th order problem is simply:
```    
d_coeffs = inv(M) G f
```

M is a (P+1)N x (P+1)N symmetric square matrix and G is a (P+1)N x Q matrix.

Generating M and G is simple. For the 0th order problem (LSAT), use:
```
M, G = lsat_transform(A, sigma, Tarr)
```
For 1st order (Lint), use:
```
M, G = lint_transform(A, sigma, Tarr)
```
For 2nd order (Quint) and higher (Pint), use:
```
M, G = noint_transform(A, sigma, Tarr, P)
```
where P is the desired polynomial order.

Once you have generated the transformation matrices for the temperature-dependent tuning curves, you can compute the decode coefficients for function *f* via:

```
import numpy as np
import np.linalg.inv as inv

d_coeffs = inv(M) @ G @ f
```
Note: *@* represents matrix multiplication in numpy.


## (V) Tranformation Matrix method for sparse solutions (i.e. splint)
We now know how to solve for decode coefficients given matrices *M* and *G*. How do we look for sparse solutions? In this section, we will talk about the L2 regularization transformation strategy. L1 regularization is under development.

The general strategy is to add a huge penalty (think 10^15 or higher) times the L2 magnitude of the decode coefficients that you want to be zero. For example, if you want a specific neuron's weight to be constant in the Lint framework, then you would penalize that neuron's d1 coefficient.

We can define a "mask" as a diagonal square matrix of size (P+1)D x (P+1)D where P is then polynomial order and D is the number of neurons. The first D diagonal entries correspond to the d0 coefficients, the next D diagonal entries corresond to d1 entries, and so on. For each coefficient that you want to be zero, set the corresponding diagonal entry in the mask to 10^15 or something huge. All other entries should be set to 0.

Once we have defined our mask, the solution for decode weights is:

```
d_coeffs = inv(M + mask) @ G @ f
```
where M and G are the transformation matrices introduced in the previous section.

This (you should check for yourself) should result in a decode coefficient vector whose components are very close to zero where you specified. From there, you can manually set these components to be exactly zero.

## (VI) Features to be added soon
* Neuron contribution and cone plots
* L1 regularization for sparse solutions
