import math
from mpmath import digamma, loggamma

def dircat(alpha, y):
    a0=sum(alpha)
    return -(math.log(alpha[y]) - math.log(a0))

def dce(alpha, y):
    a0=sum(alpha)
    return float(digamma(a0) - digamma(alpha[y]))

def nll_density(alpha, x):
    a0=sum(alpha)
    logZ = float(loggamma(a0) - sum(loggamma(a) for a in alpha))
    logp = logZ + sum((alpha[i]-1)*math.log(x[i]) for i in range(len(alpha)))
    return -logp

def example(alpha1, alpha2, y=0, C=3, s=0.1):
    # smoothed one-hot with target y
    conf=1-s
    low=s/(C-1)
    x=[low]*C
    x[y]=conf
    return {
        "dircat1": dircat(alpha1,y),
        "dircat2": dircat(alpha2,y),
        "dce1": dce(alpha1,y),
        "dce2": dce(alpha2,y),
        "nll_den1": nll_density(alpha1,x),
        "nll_den2": nll_density(alpha2,x),
    }

alpha1=[10,1,1]
alpha2=[100,10,10]