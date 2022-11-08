import numpy as np
from scipy import special

def circularPdfVonMises(th, mu, kappa, unit):

    """Returns pdf value of von Mises distribution in radians.
    
    Keyword arguments:
    th:  variable
    mu:  mean direction
    kappa: von Mises specific parameter.
    unit: determine the unit ('radian' or 'degree'). Default is radian
    
    """

    if unit == 'degree':
        th = th/180*np.pi
        mu = mu/180*np.pi

    if kappa < 0:
        raise ValueError('kappa cannot be negative...')

    if kappa == 0:
        return 1/(2*np.pi)

    th = th % (2*np.pi)
    mu = mu % (2*np.pi)

    return np.exp(kappa*np.cos(th-mu))/(2*np.pi*special.iv(0,kappa))