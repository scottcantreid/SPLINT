import numpy as np
from numpy.random import lognormal
from numpy.linalg import inv

def temp_var(rv, T, Tb = 300):
    a = -1*np.log(rv)*Tb
    rv_new = np.exp(-1*a/T)
    return rv_new

def gen_rv_raw(sigma, num_neurons):
    """Randomly generate neuron mismatch parameters

    Parameters
    ----------
    sigma : float
        sigma of the Normal distribution underlying the log-Normal distribution
    num_neurons : int
        number of neurons to generate parameters for

    Returns
    -------
    ktau : array of num_neurons mismatch paramters for tau
    km : array of num_neurons mismatch paramters for i**2
    kibg : array of num_neurons mismatch paramters for ibg
    kin : array of num_neurons mismatch paramters for iin
    """
    rv_b1m = lognormal(0.0, sigma, num_neurons)
    rv_m3 = lognormal(0.0, sigma, num_neurons)
    rv_m1 = lognormal(0.0, sigma, num_neurons)
    rv_m2 = lognormal(0.0, sigma, num_neurons)
    rv_b2m = lognormal(0.0, sigma, num_neurons)
    rv_m3 = lognormal(0.0, sigma, num_neurons)
    rv_m6 = lognormal(0.0, sigma, num_neurons)
    rv_m5 = lognormal(0.0, sigma, num_neurons)
    rv_ms1 = lognormal(0.0, sigma, num_neurons)
    rv_bs1m = lognormal(0.0, sigma, num_neurons)
    rv_bs2m = lognormal(0.0, sigma, num_neurons)
    rvList = [rv_b1m, rv_m3, rv_m1, rv_m2, rv_b2m, rv_m3, rv_m6, rv_m5, rv_ms1, rv_bs1m, rv_bs2m]
    return rvList

    return ktau, km, kibg, kin

def gen_rv_T(rvList, T, Tb = 300.0):
    newList = list()
    for rv in rvList:
        newList.append(temp_var(rv, T, Tb = Tb))

    rv_b1m = newList[0]
    rv_m3 = newList[1]
    rv_m1 = newList[2]
    rv_m2 = newList[3]
    rv_b2m = newList[4]
    rv_m3 = newList[5]
    rv_m6 = newList[6]
    rv_m5 = newList[7]
    rv_ms1 = newList[8]
    rv_bs1m = newList[9]
    rv_bs2m =newList[10]

    ktau = rv_b1m / rv_m3
    km = rv_b1m * rv_m6 / rv_m3 / rv_m5
    kibg = rv_m1 * rv_m2 / rv_b2m / rv_m3
    kin = rv_ms1 * rv_m2 * rv_b1m / rv_bs2m / rv_m3 / rv_bs1m

    return ktau, km, kibg, kin

def QIF_FI(tau, ibg, iin, ktau, km, kibg, kin):
    """Computes QIF neuron firing rates vs synaptic input currents

    Parameters
    ----------
    tau : float
        programmed time constant
    ibg : float
        programmed background current
    isyn : array
        synaptic input current values
    ktau : array
        tau mismatch parameters
    km : array
        mismatch parameters scaling i**2/2
    kibg : array
        background input current mismatch parameters
    kin : array
        synaptic input current mismatch parameters
    """
    _tau = tau * ktau
    _inp = kibg * ibg + kin * iin
    freq = np.zeros_like(ktau)
    idx = 2 * km * _inp - 1 > 0
    freq[idx] = 1./(_tau[idx] * (np.pi + 2 * np.arctan(1/np.sqrt(2 * km[idx] * _inp[idx] - 1))) /
             np.sqrt(2 * km[idx] * _inp[idx] - 1))
    return freq

def remove_bad_nrns(Araw2, iin):
    Q, N, R = Araw2.shape
    A = np.zeros_like(Araw2)
    fcut = 30.0
    bifcut = 0.75*Q
    ct = 0
    badIds = list()
    goodIds = list()
    for nrn_idx in range(N):
        inc = True
        for Tidx in [0, -1]:
            a = Araw2[:, nrn_idx, Tidx]
            if len(np.where(a > fcut)[0]) > int(Q/10.0):
                if np.diff(a)[0] < 0.1:
                    bif = iin[np.where(a < 0.01)[0][0]]
                else:
                    bif = iin[np.where(a > 0.01)[0][0]]
                if np.abs(bif) > bifcut:
                    inc = False
            else:
                inc = False
        if inc:
            A[:, :, :] = Araw2[:, :, :]
            ct += 1
            goodIds.append(nrn_idx)
        else:
            badIds.append(nrn_idx)
    Aout  = np.zeros((Q, len(goodIds), R))
    Aout[:, :, :] = A[:, np.array(goodIds), :]
    return Aout, badIds

def rv_temp(T, ktau, km, kibg, kin, Tb = 300.0):
    newList = list()
    for rv in [ktau, km, kibg, kin]:
        a = -1*np.log(rv)*Tb
        rv_new = np.exp(-1*a/T)
        newList.append(rv_new)
    return newList

def gen_A_across_T(iin, Tarr, sigma = 0.125, num_neurons = 1000, tau = 0.005, ibg = 0.5):
    rvList = gen_rv_raw(sigma, num_neurons)
    A = np.zeros((len(iin), num_neurons, len(Tarr)))
    for T_ct, T in enumerate(Tarr):
        ktau_T, km_T, kibg_T, kin_T = gen_rv_T(rvList, T)
        fnew = np.array([QIF_FI(tau, ibg, iin_val, ktau_T, km_T, kibg_T, kin_T)
                               for iin_val in iin])
        A[:, :, T_ct] = fnew
    return A

def gen_neurons(Nraw = 800, Q = 1000, sig = 0.1, R = 20, del_bad = False):
    TCarr = np.linspace(0, 50, R)
    Tarr = TCarr + 273.15
    iin = np.linspace(-1.0, 1.0, Q)
    np.random.seed(4)
    Araw1 = gen_A_across_T(iin, Tarr, num_neurons = Nraw)

    if (del_bad):
        Araw2, b = remove_bad_nrns(Araw1, iin)
    else:
        Araw2 = Araw1
        b = []

    Q, N, R = Araw2.shape

    # randomly assign half the encoding vectors to -1
    ids = np.random.permutation(np.arange(N))
    Araw3 = np.zeros_like(Araw2)
    Araw3[:, ids[:int(Nraw/2)], :] = Araw2[:, ids[:int(Nraw/2)], :]
    Araw3[:, ids[int(Nraw/2):], :] = Araw2[::-1, ids[int(Nraw/2):], :]
    A = Araw3 + np.random.normal(size = (Q, N, R))*sig

    return Tarr, A, iin

def lsat_transform(A, sigma, Tarr):
    """
    Generates matrices M and G which are used to compute LSAT weights.

    A: set of tuning curves across temperatures
    sigma: (should be 0.1 for simulated neurons) is std of noise in tuning curves
    Tarr: set of temperatures

    Matrices M and G are used in the following way to find decode weights:
    ----------------------
    M [d0] = G f
    ----------------------
    Matrices M and G have the following form:

    M = [X0]

    G = [Y0]

    Where Xn and Yn are defined as:

    Xn = <T^n * (A^T A + Z^T Z)> (<...> denotes average across temperatures)
    Yn = <T^n * A^T>
    """

    Q, n, N = A.shape

    S = N*sigma*sigma * Q * np.identity(n)

    A0 = np.zeros((n,n))

    Y0 = np.zeros((n,Q))

    for i in range(N):
        Ai = A[:,:,i]
        Ti = Tarr[i]

        A0 = A0 + Ai.T@Ai

        Y0 = Y0 + Ai.T

    X0 = A0 + S

    M = np.ndarray((n,n))
    M = X0

    G = np.ndarray((n,Q))
    G = Y0

    return M, G

def lsat(A, f, sigma, Tarr):
    """
    Implements LSAT algorithm to give d0, weights such that d(T) = d0

    A: set of tuning curves at each temperature in Tarr
    f: function we want to compute
    iin: set of input values
    sigma: std of noise for tuning curves (0.1 for sim neurons)
    Tarr: set of temperatures

    Returns d0 weights for LSAT  transform.
    """

    M, G = lsat_transform(A, sigma, Tarr)
    decc = inv(M)@G@f

    return decc

def lint_transform(A, sigma, Tarr):
    """
    Generates matrices M and G which are used to compute LinT weights.

    A: set of tuning curves across temperatures
    sigma: (should be 0.1 for simulated neurons) is std of noise in tuning curves
    Tarr: set of temperatures

    Matrices M and G are used in the following way to find decode weights:
    ----------------------
    M[d0] = G f
     [d1]
    ----------------------
    Matrices M and G have the following form:

    M = [X0 X1]
        [X1 X2]

    G = [Y0]
        [Y1]

    Where Xn and Yn are defined as:

    Xn = <T^n * (A^T A + Z^T Z)> (<...> denotes average across temperatures)
    Yn = <T^n * A^T>
    """

    Q, n, N = A.shape

    S = N*sigma*sigma * Q * np.identity(n)

    A0 = np.zeros((n,n))
    A1 = np.zeros((n,n))
    A2 = np.zeros((n,n))

    Y0 = np.zeros((n,Q))
    Y1 = np.zeros((n,Q))

    for i in range(N):
        Ai = A[:,:,i]
        Ti = Tarr[i]

        A0 = A0 + Ai.T@Ai
        A1 = A1 + Ti*Ai.T@Ai
        A2 = A2 + Ti*Ti*Ai.T@Ai

        Y0 = Y0 + Ai.T
        Y1 = Y1 + Ti*Ai.T

    T_bar = 1/N*np.sum(Tarr)
    T2_bar = 1/N*np.sum(Tarr*Tarr)

    X0 = A0 + S
    X1 = A1 + T_bar*S
    X2 = A2 + T2_bar*S

    M = np.ndarray((2*n,2*n))
    M[:n,:n] = X0
    M[n:,:n] = X1
    M[:n,n:] = X1
    M[n:,n:] = X2

    G = np.ndarray((2*n,Q))
    G[:n,:] = Y0
    G[n:,:] = Y1

    return M, G

def splint_mask(n, k_set, Omega):
    """
    Generates a regularizing mask matrix to add to M to ensure that the d1 weights are sparse.

    n: number of neurons
    k_set: a list of neuron indices for the neurons which will have nonzero d1 components
    Omega: the factor to regularize by (1e14 works pretty well)

    Returns a mask matrix to add to M which properly regularizes so that d1 is sparse.
    Also returns "zero_set", the set of indices for neurons which will have d1 set to zero.
    """
    mask = np.zeros((2*n, 2*n))
    neuron_indices = np.arange(0,n,1).astype(int)
    zero_set = np.setdiff1d(neuron_indices, k_set)
    mask[zero_set+n, zero_set+n] = Omega
    return mask, zero_set

def lint(A, f, sigma, Tarr, M = None, G = None):
    """
    Implements LinT algorithm to give d0, d1 weights such that d(T) = d0 + T * d1

    A: set of tuning curves at each temperature in Tarr
    f: function we want to compute
    iin: set of input values
    sigma: std of noise for tuning curves (0.1 for sim neurons)
    Tarr: set of temperatures

    Returns d0, d1 weights for LinT transform.
    """

    if ((M is None) | (G is None)):
        M, G = lint_transform(A, iin, sigma, Tarr)
    decc = np.linalg.inv(M)@G@f #LinT Decode Vector

    return decc

def splint(A, f, sigma, Tarr, k, Omega = 1e14, sorted_neurons = None, M = None, G = None):
    """
    Implements SpLinT algorithm to give d0, d1 weights such that d(T) = d0 + T * d1. Only k components of d1 will be nonzero.

    A: set of tuning curves at each temperature in Tarr
    f: function we want to compute
    iin: set of input values
    sigma: std of noise for tuning curves (0.1 for sim neurons)
    Tarr: set of temperatures
    k: number of nonzero components of d1 for sparse T-dependences
    Omega: regularization term

    Returns d0, d1 weights for LinT transform.

    Note: currently chooses k neurons by sorting by d0 magnitude from LinT weights. This is open to change!!
    """

    Q, n, N = A.shape
    if ((M is None) | (G is None)):
        M, G = lint_transform(A, iin, sigma, Tarr)

    if (sorted_neurons is None):
        decc = np.linalg.inv(M)@G@f #LinT Decode Vector
        d0 = decc[:n]
        d1 = decc[n:]
        sorted_neurons = np.argsort(np.abs(d0))[::-1] #sort neurons by LinT d0 magnitude

    k_set = sorted_neurons[:k]
    mask, zero_set = splint_mask(n, k_set, Omega)

    #Find SpLinT decode vectors
    decc = np.linalg.inv(M+mask)@G@f

    #Ensure that d1 values are exactly zero where they need to be
    decc[n:][zero_set] = 0.

    return decc

def quint_transform(A, sigma, Tarr):
    """
    Generates matrices M and G which are used to compute QuinT weights.

    A: set of tuning curves across temperatures
    sigma: (should be 0.1 for simulated neurons) is std of noise in tuning curves
    Tarr: set of temperatures

    Matrices M and G are used in the following way to find decode weights:
    ----------------------
    M[d0] = G f
     [d1]
     [d2]
    ----------------------
    Matrices M and G have the following form:

    M = [X0 X1 X2]
        [X1 X2 X3]
        [X2 X3 X4]

    G = [Y0]
        [Y1]
        [Y2]

    Where Xn and Yn are defined as:

    Xn = <T^n * (A^T A + Z^T Z)> (<...> denotes average across temperatures)
    Yn = <T^n * A^T>
    """

    Q, n, N = A.shape

    S = N*sigma*sigma * Q * np.identity(n)

    A0 = np.zeros((n,n))
    A1 = np.zeros((n,n))
    A2 = np.zeros((n,n))
    A3 = np.zeros((n,n))
    A4 = np.zeros((n,n))

    Y0 = np.zeros((n,Q))
    Y1 = np.zeros((n,Q))
    Y2 = np.zeros((n,Q))

    for i in range(N):
        Ai = A[:,:,i]
        Ti = Tarr[i]

        A0 = A0 + Ai.T@Ai
        A1 = A1 + Ti*Ai.T@Ai
        A2 = A2 + Ti*Ti*Ai.T@Ai
        A3 = A3 + Ti*Ti*Ti*Ai.T@Ai
        A4 = A4 + Ti*Ti*Ti*Ti*Ai.T@Ai

        Y0 = Y0 + Ai.T
        Y1 = Y1 + Ti*Ai.T
        Y2 = Y2 + Ti*Ti*Ai.T

    T_bar = 1/N*np.sum(Tarr)
    T2_bar = 1/N*np.sum(Tarr*Tarr)
    T3_bar = 1/N*np.sum(Tarr*Tarr*Tarr)
    T4_bar = 1/N*np.sum(Tarr*Tarr*Tarr*Tarr)


    X0 = A0 + S
    X1 = A1 + T_bar*S
    X2 = A2 + T2_bar*S
    X3 = A3 + T3_bar*S
    X4 = A4 + T4_bar*S

    M = np.ndarray((3*n,3*n))
    M[:n,:n] = X0
    M[n:2*n,:n] = X1
    M[2*n:,:n] = X2
    M[:n,n:2*n] = X1
    M[n:2*n,n:2*n] = X2
    M[2*n:,2*n:] = X3
    M[:n,2*n:] = X2
    M[n:2*n,2*n:] = X3
    M[2*n:,2*n:] = X4

    G = np.ndarray((3*n,Q))
    G[:n,:] = Y0
    G[n:2*n,:] = Y1
    G[2*n:,:] = Y2


    return M, G

def squint_mask(n, k_set, Omega):
    """
    Generates a regularizing mask matrix to add to M to ensure that the d1 weights are sparse.

    n: number of neurons
    k_set: a list of neuron indices for the neurons which will have nonzero d1 components
    Omega: the factor to regularize by (1e14 works pretty well)

    Returns a mask matrix to add to M which properly regularizes so that d1 is sparse.
    Also returns "zero_set", the set of indices for neurons which will have d1 set to zero.
    """
    mask = np.zeros((3*n, 3*n))
    neuron_indices = np.arange(0,n,1).astype(int)
    zero_set = np.setdiff1d(neuron_indices, k_set)
    mask[zero_set+n, zero_set+n] = Omega
    mask[zero_set+2*n, zero_set+2*n] = 300*Omega
    return mask, zero_set

def quint(A, f, sigma, Tarr, M = None, G = None):
    """
    Implements QuinT algorithm to give d0, d1, d2 weights such that d(T) = d0 + T * d1 + T^2 * d2

    A: set of tuning curves at each temperature in Tarr
    f: function we want to compute
    iin: set of input values
    sigma: std of noise for tuning curves (0.1 for sim neurons)
    Tarr: set of temperatures

    Returns d0, d1, d2 weights for LinT transform.
    """
    if ((M is None) | (G is None)):
        M, G = quint_transform(A, iin, sigma, Tarr)
    f = func(iin)
    decc = np.linalg.inv(M)@G@f #LinT Decode Vector

    return decc

def squint(A, f, sigma, Tarr, k, Omega = 1e14, sorted_neurons = None, M = None, G = None):
    """
    Implements SQuinT algorithm to give d0, d1, d2 weights such that d(T) = d0 + T * d1 +T^2 d2.
    Only k components of d1 and d2 will be nonzero.

    A: set of tuning curves at each temperature in Tarr
    f: function we want to compute
    iin: set of input values
    sigma: std of noise for tuning curves (0.1 for sim neurons)
    Tarr: set of temperatures
    k: number of nonzero components of d1 for sparse T-dependences
    Omega: regularization term

    Returns d0, d1, d2 weights for SQuinT transform.

    Note: currently chooses k neurons by sorting by d0 magnitude from LinT weights. This is open to change!!
    """

    Q, n, N = A.shape
    if ((M is None) | (G is None)):
        M, G = quint_transform(A, iin, sigma, Tarr)

    if (sorted_neurons is None):
        decc = np.linalg.inv(M)@G@f #LinT Decode Vector
        d0 = decc[:n]
        d1 = decc[n:2*n]
        d2 = decc[2*n:]
        sorted_neurons = np.argsort(np.abs(d0))[::-1] #sort neurons by d0 magnitude

    k_set = sorted_neurons[:k]
    mask, zero_set = squint_mask(n, k_set, Omega)

    #Find SpLinT decode vectors
    decc = np.linalg.inv(M+mask)@G@f

    #Ensure that d1 and d2 values are exactly zero where they need to be
    decc[n:2*n][zero_set] = 0.
    decc[2*n:][zero_set] = 0.

    return decc

def noint_transform(A, sigma, Tarr, n):
    Q, d, N = A.shape

    S = N*sigma*sigma * Q * np.identity(d)

    M = np.ndarray(((n+1)*d, (n+1)*d))
    G = np.ndarray(((n+1)*d, Q))

    for o in range(2*n+1):
        Xn = np.mean(np.power(Tarr, o))*S
        for i in range(N):
            Xn = Xn + np.power(Tarr[i], o) * A[:,:,i].T@A[:,:,i]
        j = 0
        for j in range(n+1):
            i = o - j
            if ((i > -1) & (i < n+1)):
                M[i*d:(i+1)*d, j*d:(j+1)*d] = Xn

    for o in range(n+1):
        Yn = np.zeros((d,Q))
        for i in range(N):
            Yn = Yn + np.power(Tarr[i], o)*A[:,:,i].T
        G[o*d:(o+1)*d,:] = Yn
    return M, G

def error_t(decc, A, f, Tarr, iin = None, icut = None):
    Q, n, N = A.shape
    cut = np.array([True]*Q)
    if (icut is not None):
        cut = (iin > icut[0]) & (iin < icut[1])
        Q = len(iin[cut])
    order = int(len(decc)/n)
    err = np.ndarray(N)
    for i in range(N):
        d = np.zeros(n)
        for k in range(order):
            d = d + np.power(Tarr[i], k)*decc[k*n:(k+1)*n]
        f_decode = A[:,:,i]@d
        error = np.sqrt(np.sum((f_decode - f)[cut]**2)/Q)
        err[i] = error
    return err

"""
Computing Least Squares d_star at temperature Tarr[t]
"""
def ls_dstar(A, f, sigma, t):
    Q, n, N = A.shape
    A = A[:,:,t]

    Left = Q*sigma**2*np.identity(n) + np.dot(A.T, A)
    Left = np.linalg.inv(Left)
    Right = np.dot(A.T, f)
    dstar_LS = np.dot(Left, Right)
    return dstar_LS
