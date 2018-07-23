# T-Dependent-NEF
Temperature robustness can be achieved in mixed analog/digital computers in the neuromorphic NEF framework by changing neural weights as a function of temperature.

How to use this package:

## (I) Simulating neuron tuning curves:
    Use gen_neurons(Nraw, Q, sig, R, del_bad=False) to generate temperature dependent tuning curves of Nraw neurons over a domain of Q input  at R different temperatures. del_bad=True will lead to a slightly smaller neuron yield, as "bad" neurons which rarely fire will be cut out of the neural population. del_bad=True is not recommended.
    
      Example:
      Tarr, A, iin = tempy.gen_neurons(Nraw = 5000, Q = 1000, sig = 0.1, R = 30)
    
## (II) The quick and easy way of solving for optimal decode vectors for function f:
   ### (A) Least Squares decode vector centered at temperature Tarr[t]
    
          Example:
          d_ls = tp.ls_dstar(A, f, sigma, t = t)
        
   ### (B) LSAT decode vector
        There are multiple ways to doing this, but the simplest is to use the function lsat()
        
          Example:
          d_lsat = (A, f, sigma, Tarr)
          
        If you want to compute decode vectors many times, it is recommended that you use the transfomration matrix method, which is explained later.
        
  ###  (C) Lint Decode Vector
        Use d_lint = lint(A, f, sigma, Tarr).
        
   ### (D) Splint Decode Vector
        Use d_splint = splint(A, f, sigma, Tarr, k) where k is the number of neurons which will have linear-in-temperature weights. The default is to sort neurons by the magnitude of their lint d_0 weights. If you want to use a custom neuron sort, input it as parameter "sorted_neurons." For more control, it is highly recommended to use the transformation matrix method.
        
   ### (E) Quint 
        Use d_quint = quint(A, f, sigma, Tarr)
        
   ### (F) Squint
        You could choose to use squint(A, f, sigma, Tarr, k) where k is the number of neurons with quadratic weights. However, it is recommended to use the transformtion matrix method, as the optimal way to choose a sparse population for quint weights is still an open question.
        
## (III) Transformation matrix method for non-sparse solutions
    LSAT, Lint, Quint, and polynomial-order Pint weights can all be unified in a simple mathematical framework which characterizes a linear transformation from R^Q, the discretized function space, to R^((P+1)N), the space of P^th polynomial order decode weight coefficients.
    A temperature-dependent decode vector may be expressed as a polynomial series summation:
    
      d(T) = d_0 + T d_1 + T^2 d_2 + ... + T^P d_P
    
    Each coefficient d_i is a vector of length N, where N is the number of neurons in the population. We can create a stacked vector 
    D = [d_0, d_1, ..., d_P]
    
    Defined in this way, the solution to the P^th order problem is simply:
    
    D = inv(M) G f
    
    M is a (P+1)N x (P+1)N symmetric square matrix and G is a (P+1)N x Q matrix.
    
    Generating M and G is simple. For the 0th order problem (LSAT), use:
      M, G = lsat_transform(A, sigma, Tarr)
    For 1st order (Lint), use:
      M, G = lint_transform(A, sigma, Tarr)
    For 2nd order (Quint) and higher (Pint), use:
      M, G = noint_transform(A, sigma, Tarr, n), where n is the desired polynomial order.
      
      
        
        
    

