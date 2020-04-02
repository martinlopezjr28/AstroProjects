import numpy as np

# ====================================================#
# Define physical constants

class Constants():
    def __init__(self):
        self.msun = 1.989e33
        self.rsun = 6.955e10
        self.G  = 6.674e-8
        self.yr = 3.1536e7
        self.h  = 6.6260755e-27
        self.kB = 1.380658e-16
        self.mp = 1.6726219e-24
        self.me = 9.10938356e-28
        self.c  = 2.99792458e10
        self.pc = 3.085677581e18
        self.au = 1.496e13
        self.q = 4.8032068e-10
        self.eV = 1.6021772e-12
        self.sigmaSB = 5.67051e-5
        self.sigmaT = 6.6524e-25

        print ("Constants defined...")
        return None

c = Constants()

global Ro, Mo
Ro=6.96e10
Mo=1.99e33
G=6.6726e-08

def Orbit(e,r,rperi,Mbin,Mstar):
    mu = G * (Mbin + Mstar)
    k = mu * rperi * (1 + e)**(1./2.)
    f = np.arccos((k**2 / (mu * r) - 1)/e)
    v = np.sqrt(2) * ((mu**2 * (e**2 - 1))/ k**2 + mu/r)**(1./2.)

    return v,k,f