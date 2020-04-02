import numpy as np
import matplotlib.pyplot as pl
from astropy.table import Table
from astropy.io import ascii
import seaborn as sns
sns.set_style('ticks')
from scipy.integrate import odeint
from mpl_toolkits.mplot3d import Axes3D
from fractions import Fraction
import BBHorbit as bhb


G = 6.67e-8
Mo = 1.989e33
AU = 1.49597870e+13
a = 800*AU
ecc = 0.75
Ro=6.96e10
day = 60*60*24
hrs = 60*60
tdyn = np.sqrt(Ro**3/(G*Mo))

#In solar units from (Tout et. al 1996)
def MtoR(M):
    theta = 1.71535900
    ell = 6.59778800
    kappa = 10.0885500
    lmbda = 1.01249500
    mu = 0.07490166
    nu = 0.01077422
    xi = 3.08223400
    o = 17.84778000
    pi = 0.00022582

    R = (theta * M**(2.5) + ell * M**(6.5) + kappa * M**(11.0) + lmbda * M**(19.0) + mu * M**(19.5)) / (nu + xi * M**(2.0) + o * M**(8.5) + M**(18.5) + pi * M**(19.5))

    return R

def find_nearest(array,value):
    '''
    Returns index of nearest number to value in array
    '''
    idx = (np.abs(array-value)).argmin()
    return idx

def tidal(Mbh,Mstar,Rstar):
    """
    *All quantities should be in CGS*

    Calculates the tidal radius of an object (black hole) for a given star.
    The parameter names are specified for black holes but this function is general for any tidal radius.

    Parameters:
    Mbh = The mass of the object you want the tidal radius for [g] (scalar)
    Mstar = The mass of the star you want to tidally disrupt [g] (scalar)
    Rstar = The radius of the star you want to tidally disrupt [cm] (scalar)

    Return:

    The tidal radius [cm] (scalar)

    """

    return Rstar*(Mbh/Mstar)**(1./3)

def c_rad(Mc,vel):
    circ_rad = G*Mc/(vel)**2
    """
    *All quantities should be in CGS*

    Calculates the radius of a circular orbit of an object with speed vel around a central mass.

    Parameters:
    Mc = The central mass being orbited [g] (scalar)
    vel = The velocity of the orbiting object


    Return:

    circ_rad = The circularization radius of the orbit [cm] (scalar)
    """
    return circ_rad

def inside_tide(R_star, R_closest, R_tau):
    """
    *All quantities should be in CGS*

    Calculates and prints the percentage of a star with a given pericenter inside a given tidal radius

    Parameters:
    R_star = The radius of the star being disrupted [cm] (scalar)
    R_closest = The pericenter of the orbit of the star [cm] (scalar)
    R_tau = The tidal radius of the object that the star is orbiting [cm] (scalar)

    Return:
    None
    """

    Dstar = 2.0 * R_star
    R_IB = R_closest - R_star #This is the innermost radius of the star to the BH
    inside = R_tau - R_IB #Amount of star inside the tidal radius in solar units
    perc_in = round(inside/Dstar * 100,2)

    if perc_in > 100:
        perc_in = 100.0
    elif perc_in < 0:
        perc_in = 0.0

    print('\n' + str(perc_in) + '\% of the star is inside of the tidal radius')
    if R_IB < 0:
        print('\nThis run results in a physical collision, ' + str(round(abs(R_IB)/Dstar * 100,2)) + '\% of the star has hit the BH')

def get_first_min(sol_array, this_min, BH):

    if BH == 1.:
        sol_array = sol_array['r3_1']
    elif BH == 2.:
        sol_array = sol_array['r3_2']

    #This bool list will have the booleans of each minimum of the distance to BH1
    bool_list = []

    #creates a new list to modify
    new_array = sorted(sol_array)

    #keeps the while loop going as it finds more minimums
    more_min = True

    i = 0

    while i < len(new_array)/1000:
        #this is boolen the current minimum in the sol_array array
        curr_min_bool = find_nearest(sol_array,new_array[i])

        #These conditions ensure the min of the array is an actual minimum
        if sol_array[curr_min_bool] - sol_array[curr_min_bool + 1] < 0:
            if sol_array[curr_min_bool] - sol_array[curr_min_bool - 1] < 0:
                bool_list.append(curr_min_bool)
        i += 1

    #chooses the correct index in the array
    min_index = this_min - 1

    #sorts the boolean list of minimums into ascending order
    new_bool_list = sorted(bool_list)

    #gives back the desired min
    chosen_min_bool = new_bool_list[min_index]

    print('\nFor BH ' + str(BH))
    print('-------------------------')
    print('\nNumber of minimums:', len(bool_list))
    print('\nMinimum number ' + str(this_min) + ' has been chosen')
    print('\nChosen minimum is:', str(round(sol_array[chosen_min_bool]/Ro,2)),
          ' Ro')
    print('**************************')

    return chosen_min_bool

def make_coord_table(sol_array, distance, Rstar, rt1, rt2, rtc, which_BH = 1., which_min_1 = 1, which_min_2 = 1, tfactor = 5./6. , energy = False, get_table = False, write_out = False,
    a = 0, m1 = 0, m2 = 0, m3 = 0, r = 0, rperi = 0, v_inf = 0):

    #This will get the boolean when the star is closest to BH1
    peri_BH1_bool = get_first_min(sol_array, which_min_1, BH = 1)
    peri_BH2_bool = get_first_min(sol_array, which_min_2, BH = 2)

    peri_BH1 = sol_array['r3_1'][peri_BH1_bool]
    peri_BH2 = sol_array['r3_2'][peri_BH2_bool]

    if which_BH == 1.:
        use_array = sol_array['r3_1']
        rtau = rt1
        peri_bool = peri_BH1_bool
        peri = peri_BH1

    elif which_BH == 2.:
        use_array = sol_array['r3_2']
        rtau = rt2
        peri_bool = peri_BH2_bool
        peri = peri_BH2

    #This will get the time of periaps for the star
    t_peri_BH = sol_array['t'][peri_bool]

    #This bool will pick out the times I want to look through for chosen distance
    real_be4_peri_bool = np.logical_and(sol_array['t'] < t_peri_BH, sol_array['t'] > (t_peri_BH * (tfactor)))

    #This bool will give me the true length of the times before peri
    fake_be4_peri_bool = sol_array['t'] < t_peri_BH

    #sets the distance to BH1 that you want
    desired_r = distance

    #This will give the true boolean of the wanted coordinates because the truncated boolean would give the wrong answer
    true_length = len(use_array[fake_be4_peri_bool])
    truncated_length = len(use_array[real_be4_peri_bool])

    #This will give me how much to add to the index of the desired r bool
    difference = true_length - truncated_length

    desired_r_bool = find_nearest(use_array[real_be4_peri_bool], desired_r) + difference

    #This is the actual distance in the code to the BH
    desired_dist_2_BH = use_array[desired_r_bool]

    print('\nThe desired starting distance to BH', str(int(which_BH)),'=',
          str(round(distance/Ro,2)), 'Ro =', str(round(distance/rtc,2)), 'rtc')
    print('\nThe actual starting distance to BH', str(int(which_BH)), '=',
          str(round(desired_dist_2_BH/Ro,2)), 'Ro =',
          str(round(desired_dist_2_BH/rtc,2)), 'rtc')

    sx = sol_array['x3'][desired_r_bool]
    sy = sol_array['y3'][desired_r_bool]
    sz = sol_array['z3'][desired_r_bool]

    svx = sol_array['vx3'][desired_r_bool]
    svy = sol_array['vy3'][desired_r_bool]
    svz = sol_array['vz3'][desired_r_bool]

    BH1x = sol_array['x1'][desired_r_bool]
    BH1y = sol_array['y1'][desired_r_bool]
    BH1z = sol_array['z1'][desired_r_bool]

    BH1vx = sol_array['vx1'][desired_r_bool]
    BH1vy = sol_array['vy1'][desired_r_bool]
    BH1vz = sol_array['vz1'][desired_r_bool]

    BH2x = sol_array['x2'][desired_r_bool]
    BH2y = sol_array['y2'][desired_r_bool]
    BH2z = sol_array['z2'][desired_r_bool]

    BH2vx = sol_array['vx2'][desired_r_bool]
    BH2vy = sol_array['vy2'][desired_r_bool]
    BH2vz = sol_array['vz2'][desired_r_bool]

    coord = np.array([BH1x, BH1y, BH1z, BH1vx, BH1vy, BH1vz, BH2x, BH2y, BH2z, BH2vx, BH2vy, BH2vz,sx, sy, sz, svx, svy, svz])
    col_names = ['BH1x', 'BH1y', 'BH1z', 'BH1vx', 'BH1vy', 'BH1vz', 'BH2x', 'BH2y', 'BH2z', 'BH2vx', 'BH2vy', 'BH2vz','sx', 'sy', 'sz', 'svx', 'svy', 'svz']
    t_coord = Table(coord,names=col_names)

    print('\nChosen closest aproach to 1 (solar radii)')
    print(round(peri_BH1/Ro,3))

    print('\nChosen closest aproach to 2 (solar radii)')
    print(round(peri_BH2/Ro,3))

    print('\nPeri time:', round(t_peri_BH/tdyn,2), '[t_dyn] =',
           round(t_peri_BH/day,2),'[days]')
    print(str(round(tfactor,2)),'Peri time:',
        round(t_peri_BH*(tfactor)/tdyn,2), '[t_dyn] =',
        round(t_peri_BH*(tfactor)/day,2), '[days]')
    print('Time of desired distance:',
        round((sol_array['t'][desired_r_bool])/tdyn,2), '[t_dyn] =',
        round((sol_array['t'][desired_r_bool])/day,2), '[days]')


    inside_tide(Rstar,peri,rtau)

    if energy == True:

        vx_peri1 = sol_array['vx3'][peri_BH1_bool]
        vx_peri2 = sol_array['vx3'][peri_BH2_bool]

        vy_peri1 = sol_array['vy3'][peri_BH1_bool]
        vy_peri2 = sol_array['vy3'][peri_BH2_bool]

        vz_peri1 = sol_array['vz3'][peri_BH1_bool]
        vz_peri2 = sol_array['vz3'][peri_BH2_bool]

        v_peri1 = np.sqrt(vx_peri1**2 + vy_peri1**2 + vz_peri1**2)
        v_peri2 = np.sqrt(vx_peri2**2 + vy_peri2**2 + vz_peri2**2)

        r_peri1 = sol_array['r3_1'][peri_BH1_bool]
        r_peri2 = sol_array['r3_2'][peri_BH2_bool]

        KE_peri1 = 0.5 * m3 * v_peri1**2
        KE_peri2 = 0.5 * m3 * v_peri2**2

        PE_peri1 = -(G * m1 * m3)/r_peri1
        PE_peri2 = -(G * m2 * m3)/r_peri2

        E_peri1 = PE_peri1 + KE_peri1
        E_peri2 = PE_peri2 + KE_peri2

        print('\n-------------------------')
        print('-------------------------')
        print('Energy of star in BH1 orbit is:','{:0.2e}'.format(E_peri1),
              '[ergs]')
        print('Energy of star in BH2 orbit is:','{:0.2e}'.format(E_peri2),
               '[ergs]')
        print('-------------------------')
        print('-------------------------')

    if write_out:
        #Converts from CGS to code units
        a_code = str(int(round(a / Ro))) + 'a'
        mBH1_code = str(int(m1 / Mo)) + 'M'
        mBH2_code = str(int(m2 / Mo)) + 'M'
        mstar_code = str(int(m3 / Mo)) + 'M'
        rstar_code = str(int(Rstar / Ro)) + 'R'
        r_code = str(int(r/Ro)) + 'r'
        rp_code = str(int(rperi/Ro)) + 'rp'
        desired_dist_2_BH_code = str(int(desired_dist_2_BH/Ro)) + 'R'
        v_inf_code = str(int(v_inf/(1e5))) + 'kms'

        t_name = mstar_code + rstar_code + v_inf_code + '-' + mBH1_code + mBH2_code +'_' + r_code + rp_code + desired_dist_2_BH_code + '.dat'
        path_name = 'IC_Coord/' + t_name

        print('\nWriting to ' + path_name)

        ascii.write(t_coord,path_name)

    if get_table:
        return t_coord
