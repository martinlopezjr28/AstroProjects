import numpy as np
from astropy.io import ascii
from astropy.table import Table, Column


G = 6.6726e-08
Ro = 6.96e10
Mo = 1.99e33
c = 2.9979e+10
day = 60*60*24

#Define units

DistUnit = Ro
MassUnit = Mo
TimeUnit = np.sqrt(DistUnit**3/(G*MassUnit))
VelUnit = DistUnit/TimeUnit
AngMomUnit = DistUnit*VelUnit*MassUnit

SpinUnit = AngMomUnit*c/(G*Mo**2)
Tday = TimeUnit/(60*60*24)

#=====================================================
# Finds index of the element in an array
# with the closest value to "value"

def find_nearest(array,value):
        idx = (np.abs(array-value)).argmin()
#     return array[idx]
        return idx

def Get_dynamics(filename):

    dyn = np.genfromtxt(filename)

    colnames = ('t','Macc_bh','Engy_bh','PaccX_bh','PaccY_bh','PaccZ_bh','LaccX_bh','LaccY_bh' \
                ,'LaccZ_bh','M_gas','X_gas','Y_gas','Z_gas','PX_gas','PY_gas','PZ_gas' \
                ,'LX_gas','LY_gas','LZ_gas', 'M_star','X_star','Y_star','Z_star','PX_star' \
                ,'PY_star','PZ_star','LX_star','LY_star','LZ_star','M_bh','X_bh','Y_bh' \
                ,'Z_bh','PX_bh','PY_bh','PZ_bh','LX_bh','LY_bh','LZ_bh','Macc_star', 'Engy_star' \
                ,'PaccX_star','PaccY_star','PaccZ_star','LaccX_star','LaccY_star' \
                ,'LaccZ_star','LaccX_starCM','LaccY_starCM','LaccZ_starCM','LaccX_bhCM' \
                ,'LaccY_bhCM','LaccZ_bhCM','rp')

    print np.shape(dyn), len(colnames)
    dat = Table(dyn,names=colnames)


    return dat

def spin_disk(M,Mf):

    af = (2./3)**(0.5) * (M/Mf)*(4 - np.sqrt(18*(M/Mf)**2 - 2))

    return af

def get_Ltot(dyn,withacc = False):
    L_totx = dyn['LX_bh']+dyn['LX_star']+dyn['LX_gas']
    L_toty = dyn['LY_bh']+dyn['LY_star']+dyn['LY_gas']
    L_totz = dyn['LZ_bh']+dyn['LZ_star']+dyn['LZ_gas']

    L_totxa = dyn['LX_bh']+dyn['LX_star']+dyn['LX_gas'] + dyn['LaccX_bh']
    L_totya = dyn['LY_bh']+dyn['LY_star']+dyn['LY_gas'] + dyn['LaccY_bh']
    L_totza = dyn['LZ_bh']+dyn['LZ_star']+dyn['LZ_gas'] + dyn['LaccZ_bh']

    L_tot = np.sqrt(L_totx**2 + L_toty**2 + L_totz**2)
    L_totac = np.sqrt(L_totxa**2 + L_totya**2 + L_totza**2)


    if withacc:
        return L_totac
    else:
        return L_tot

def get_3D_arrays(dyn,ptype='BH'):

    pos = np.zeros((len(dyn['t']),3))
    vel = np.zeros((len(dyn['t']),3))

    if ptype =='BH':
        pos[:,0] = dyn['X_bh']
        pos[:,1] = dyn['Y_bh']
        pos[:,2] = dyn['Z_bh']
        vel[:,0] = dyn['PX_bh']/dyn['M_bh']
        vel[:,1] = dyn['PY_bh']/dyn['M_bh']
        vel[:,2] = dyn['PZ_bh']/dyn['M_bh']

    elif ptype =='Star':
        pos[:,0] = dyn['X_star']
        pos[:,1] = dyn['Y_star']
        pos[:,2] = dyn['Z_star']
        vel[:,0] = dyn['PX_star']/dyn['M_star']
        vel[:,1] = dyn['PY_star']/dyn['M_star']
        vel[:,2] = dyn['PZ_star']/dyn['M_star']

    return pos,vel

def Get_BinCM(pos1,pos2,vel1,vel2,m1,m2):

    print np.shape(pos1)

    m_1 = np.reshape(m1,(len(m1),1))
    m_2 = np.reshape(m2,(len(m2),1))

#     print np.shape(m1), np.shape(m_1)
    pos_CM = (pos1*m_1 + pos2*m_2)/(m_1+m_2)
    vel_CM = (vel1*m_1 + vel2*m_2)/(m_1+m_2)

    return pos_CM,vel_CM

def rel_2CM(pos,vel,pos_CM,vel_CM,ptype='BH'):

    dist_2CM = np.linalg.norm((pos-pos_CM),axis=1)
    vel_2CM = np.linalg.norm((vel-vel_CM),axis=1)
#     r_xy = np.linalg.norm(pos[:,:2],axis=1)
    return dist_2CM,vel_2CM

def normal_vec(pos,pos_CM,vel,vel_CM):

    pos_2CM = pos-pos_CM
    vel_2CM = vel-vel_CM

    j_orb = np.cross(pos_2CM,vel_2CM)
#     print np.shape(j_orb)
    j_norm = np.reshape(np.linalg.norm(j_orb,axis=1),(len(j_orb),1))
#     print np.shape(j_norm)


    return j_orb/j_norm


def get_L(dyn,ptype='BH',withacc = False):


    Lacc = np.zeros((len(dyn['t']),3))

    if ptype =='BH':
        Lacc[:,0] = dyn['LaccX_bh']
        Lacc[:,1] = dyn['LaccY_bh']
        Lacc[:,2] = dyn['LaccZ_bh']
        Macc = dyn['M_bh']

    elif ptype =='Star':
        Lacc[:,0] = dyn['LaccX_star']
        Lacc[:,1] = dyn['LaccY_star']
        Lacc[:,2] = dyn['LaccZ_star']
        Macc = dyn['M_star']


    dLacc = np.diff(Lacc,axis=0)
    dMacc = np.reshape(np.diff(Macc),(len(np.diff(Macc)),1))
    jacc = dLacc/dMacc

#     dLacc_x = np.diff(Lacc[:,0])
#     dLacc_y = np.diff(Lacc[:,1])
#     dLacc_z = np.diff(Lacc[:,2])
#     dMacc = np.diff(Macc)

#     jacc = np.zeros((len(dMacc),3))
#     jacc[:,0] = dLacc_x/dMacc
#     jacc[:,1] = dLacc_y/dMacc
#     jacc[:,2] = dLacc_z/dMacc

    return jacc,dMacc,Lacc,Macc



def get_spin(dyn,ptype='BH'):

    global SpinUnit

    jacc,dMacc,Lacc,Macc = get_L(dyn,ptype)
    Lacc_n = np.linalg.norm(Lacc,axis=1)

    return Lacc_n*SpinUnit/(Macc)**2,Lacc/np.reshape(Lacc_n,(len(Lacc_n),1))



def get_spinangle(dyn):

#     jacc1,dMacc1,Lacc1,Macc1 = get_L(dyn,'BH')
#     jacc2,dMacc2,Lacc2,Macc2 = get_L(dyn,'Star')

    # Get 3D arrays pos and vel
    posBH,velBH = get_3D_arrays(dyn,'BH')
    posStar,velStar = get_3D_arrays(dyn,'Star')

    # Get CM pos and vel
    pos_CM,vel_CM = Get_BinCM(posBH,posStar,velBH,velStar,dyn['M_bh'],dyn['M_star'])

    # Get distance and velocities relative to CM
    dist_BHCM, vel_BHCM = rel_2CM(posBH,velBH,pos_CM,vel_CM)
    dist_StarCM, vel_StarCM = rel_2CM(posStar,velStar,pos_CM,vel_CM)

    # Get vector normal to orbital plane
    j_orb = normal_vec(posBH,pos_CM,velBH,vel_CM)

    # Get normalized spin vector
    print 'Getting spins'
    a_BH,L_BH = get_spin(dyn,'BH')
    a_Star,L_Star = get_spin(dyn,'Star')

#     print np.shape(L_BH),np.shape(np.transpose(j_orb))
    print '\n Getting angle between spin and J_orb'
    ddot_BH = (L_BH[:,0]*j_orb[:,0] + L_BH[:,1]*j_orb[:,1] + L_BH[:,2]*j_orb[:,2])
    ddot_Star = (L_Star[:,0]*j_orb[:,0] + L_Star[:,1]*j_orb[:,1] + L_Star[:,2]*j_orb[:,2])

#     angle_BH = np.arccos(ddot_BH)
#     angle_Star = np.arccos(ddot_Star)
    angle_BH = (ddot_BH)
    angle_Star = (ddot_Star)

    return angle_BH, angle_Star


def rj_isco(a_bh):
    z1 = 1 + pow(1-a_bh*a_bh,1./3.) * (pow(1+a_bh,1./3.) + pow(1-a_bh,1./3.))
    z2 = pow(3*a_bh*a_bh + z1*z1,1./2.)
    rms_i = (3 + z2 - np.sqrt((3-z1)*(3+z1+2*z2)))
    jcrit = 2/(pow(3.,3./2.)) * ( 1 + 2*np.sqrt(3.*rms_i -2. ) )

    return rms_i, jcrit



