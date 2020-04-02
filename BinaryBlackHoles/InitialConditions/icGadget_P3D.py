# =====================================================
#  Make hdf5 file with SPH initial conditions for GADGET
#  Specialized for polytropic stars
#  !INFORMATION AT END OF MODULE!
# =====================================================

# ==========================================================#
#  Import packages and modules needed

import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as pl
from astropy.io import ascii
from astropy.table import Table, Column
import time

import NestedPolyStar as nps  # Used to obtain polytropic stars
import SphericalSPHDist as sph # Used to build spherical particle distributions
import dfunc as df #Used to create directories after creating initial conditions
import BinaryBHOrbit as bhb #Used for binary initial conditions

# Time the execution of the program
start = time.time()

# ====================================================#
# Define physical constants
global Ro, Mo
Ro=6.96e10
Mo=1.99e33
G=6.6726e-08
AU = 1.496e13

#Instantiating all variables needed to create paths
Mstar = None
Rstar = None
n = None
n1 = None
n2 = None
mBH = None
BHx = None 
BHy = None 
BHz = None 
BHvx = None 
BHvy = None 
BHvz = None
mBH1 = None
mBH2 = None
e = None
i = None
P = None
sep_at_peri = None
roffset = None
rperi = None
e_orb = None
BH1x = None
BH2x = None
BH1vx = None
BH1vy = None
BH2vx = None
BH2vy = None
x_orbit = None
y_orbit = None
z_orbit = None
vx_orbit = None
vy_orbit = None
vz_orbit = None
DistUnit = None
MassUnit = None
VelUnit = None

# ======================================================#
# Define Units
DistUnit = Ro
MassUnit = Mo
TimeUnit = np.sqrt(DistUnit**3 /(G*MassUnit))
DensUnit = MassUnit/DistUnit**3
VelUnit = DistUnit/TimeUnit
E_perMassUnit = VelUnit**2
P_Unit = E_perMassUnit*DensUnit



# ====================================================#
#                                                     #
#   Define outputfile, number of particles needed     #
#   and properties of simulation                      #
#                                                     #
# ====================================================#

# ======================================================#
# Define number of SPH particles

N_p = int(1e5)  # number of SPH particles wanted

Npstring = str(N_p)
N_k = len(Npstring) - 4
N_p_code = Npstring[0]+'0'*N_k+'k'

# ===================================================
# Main Components of the simulation

addPoly = True
addNestedPoly = False
addOneBH = False
addTwoBH = True
set_coord = True
starpart = True
scale_to_units = True

if set_coord:
    coord_path = '../IC_Coord/1M43R30kms-15M15M_1336r668rp801R.dat'
    coord_table = ascii.read(coord_path)
    coord_array = [coord_table[0][0], coord_table[0][1], coord_table[0][2], coord_table[0][3], coord_table[0][4],
                   coord_table[0][5], coord_table[0][6], coord_table[0][7], coord_table[0][8], coord_table[0][9],
                   coord_table[0][10], coord_table[0][11], coord_table[0][12], coord_table[0][13], coord_table[0][14],
                   coord_table[0][15], coord_table[0][16], coord_table[0][17]]

if addPoly or addNestedPoly:
    Mstar = 1.0 * MassUnit
    Rstar = 43.0 * DistUnit

    # ======================================================#
    # Define positions for SPH particles (in CGS)
    Sx = -250. * DistUnit
    Sy = 0. * DistUnit
    Sz = 0. * DistUnit

    # ======================================================#
    # Define velocities for SPH particles (in CGS)
    Svx = 150. * 1e5
    Svy = 0. * 1e5
    Svz = 0. * 1e5

    if set_coord:
        Sx = coord_array[12]
        Sy = coord_array[13]
        Sz = coord_array[14]

        Svx = coord_array[15]
        Svy = coord_array[16]
        Svz = coord_array[17]

    prel = np.array([Sx,Sy,Sz]) 
    vrel = np.array([Svx,Svy,Svz])

    vrot = []
    vrad = []

if addPoly:
    n = 1.5

if addNestedPoly:
    n1 =3.0 #Core Polytropic Index
    n2 = 2.135 #Envelope Polytropic Index

if addOneBH:
    #Mass of the black hole
    mBH = 15.0 * MassUnit

    #BH Position in CGS
    BHx = 0.0 * DistUnit
    BHy = 0.0 * DistUnit
    BHz = 0.0 * DistUnit

    #BH velocity in CGS
    BHvx = 0.0 * 1e5
    BHvy = 0.0 * 1e5
    BHvz = 0.0 * 1e5

    if set_coord:
        BHx = coord_array[0]
        BHy = coord_array[1]
        BHz = coord_array[2]

        BHvx = coord_array[3]
        BHvy = coord_array[4]
        BHvz = coord_array[5]

    # BH properties    
    posBH = np.array([BHx,BHy,BHz])
    velBH = np.array([BHvx,BHvx,BHvx])


if addTwoBH:
    #Mass quantities of the black holes
    mBH1 = 15.0 * MassUnit
    mBH2 = 15.0 * MassUnit

    orb = bhb.ICs(mBH1,mBH2,Mstar)
    rtc = orb.tidal(mBH1+mBH2, Mstar, Rstar)

    #Binary Orbit Properties
    e = 0.5
    sep_at_peri = 0.1 * AU

    if addPoly:

        if set_coord == True:
            rfact = sep_at_peri
        else:
            rfact = rtc

        #Stellar Orbital Properties
        roffset =  10.0 * rtc #Distance from the BHb
        rperi =  5.0 * rtc #Closest approach to BHb
        e_orb = 1.0 #We want to send it on a parabolic orbit
        i = np.pi/4 #Usually wants this to be non-zero
        pointto = 'm1'
        v_inf = 30.0 * (1.0e5)

    if addPoly:
        initialvalues, P = orb.get_IC(mBH1, mBH2, sep_at_peri, e, Mstar, roffset, rperi, e_orb, i, pointto, v_inf)
    else: 
        initialvalues = orb.getBinary(mBH1,mBH2,sep_at_peri,e)
        P = initialvalues[8]

    if set_coord:
        initialvalues = coord_array

    #Position of BH1
    BH1x = initialvalues[0]
    BH1y = initialvalues[1]
    BH1z = initialvalues[2]

    #Velocity of BH1
    BH1vx = initialvalues[3]
    BH1vy = initialvalues[4]
    BH1vz = initialvalues[5]

    #Position of BH2
    BH2x = initialvalues[6]
    BH2y = initialvalues[7]
    BH2z = initialvalues[8]

    #Velocity of BH2
    BH2vx = initialvalues[9]
    BH2vy = initialvalues[10]
    BH2vz = initialvalues[11]

    #Final position arrays of BH's in CGS
    posBH1 = np.array([BH1x,BH1y,BH1z])
    posBH2 = np.array([BH2x,BH2y,BH2z])

    #Final velocity arrays of BH's in CGS
    velBH1 = [BH1vx, BH1vy, BH1vz]
    velBH2 = [BH2vx, BH2vy, BH2vz]

    if addPoly:
        #position/velocity of the star from the origin in CGS
        x_orbit = initialvalues[12]
        y_orbit = initialvalues[13]
        z_orbit = initialvalues[14]

        vx_orbit = initialvalues[15]
        vy_orbit = initialvalues[16]
        vz_orbit = initialvalues[17]

        prel = np.array([x_orbit,y_orbit,z_orbit])
        vrel = np.array([vx_orbit,vy_orbit,vz_orbit])

# ====================================================#
# Apply random rotation to healpix shells

rotshell = True   # Turn on / off

# ====================================================#
# Apply Gaussian distribution to shell's radius

gaussRad = False  # Turn on / off
dr_sigma = 0.1 # 1 sigma of gaussian Dist will be within 0.1 of shell's width
Nsigma = 3.0   # Random Gaussian distribution up to 3 Sigma

# ===========================================================
# Replace gas particles in the core by point mass and a shell

replace_core = False

if replace_core:
    coretype = 'p_core'
else:
    coretype = 's_core'


# ===========================================================
# Instantiate the class for directory functions
direc = df.Create_Directory(N_p_code, DistUnit, MassUnit, VelUnit, Mstar, Rstar, n, n1, n2, mBH, Sx, Sy, Sz, Svx, Svy, Svz, BHx, BHy, BHz, BHvx, BHvy, BHvz,
                            mBH1, mBH2, BH1x, BH2x, BH1vx, BH1vy, BH2vx, BH2vy, e, P, sep_at_peri, roffset, rperi, e_orb, i, x_orbit, y_orbit, z_orbit, vx_orbit,
                            vy_orbit, vz_orbit, addPoly, addNestedPoly, addOneBH, addTwoBH)

# ===========================================================
# Create the directory paths and return file names
Filename, valname, DATfilename = direc.create_IC(addPoly,addNestedPoly,addOneBH,addTwoBH)

# ===========================================================
#This is for when I add a star particle to the simulation instead of two BHs
if starpart == True:
    Filename = Filename[:-5] + '_starpart.hdf5'
    valname =  valname[:-4] + '_starpart.txt'
    DATfilename = DATfilename[:-4] + '_starpart.dat'

f = open(valname,'w')
    
# ==========================================================#
# ====================================================#
#  Here we call the functions
# ====================================================#
# ==========================================================#

if addPoly:

    # ====================================================#
    #  First we obtain a Polytrope

    r,rho,u,P,M = nps.Get_Polytrope(Mstar,Rstar,n)

    # ===================================================================
    # this Table is used later to create the SPH particle distribution
    # ===================================================================
    dat = Table([r,rho,u,P,M],names=('r', 'rho', 'u', 'P', 'M'))
    ascii.write(dat,DATfilename)

    if scale_to_units:
        dat_solar = Table([r/DistUnit,rho/DensUnit,u/E_perMassUnit,P/P_Unit,M/MassUnit],names=('r', 'rho', 'u', 'P', 'M'))
        ascii.write(dat_solar, DATfilename[:-4] + '_scaled.dat')


    # ====================================================#
    # Get interpolated profiles to build SPH star
    # ====================================================#

    rho_int = interp1d(dat['r'],dat['rho'],bounds_error=False, fill_value=dat['rho'][-1])
    u_int = interp1d(dat['r'],dat['u'],bounds_error=False, fill_value=dat['u'][0])
    M_int = interp1d(dat['r'],dat['M'],bounds_error=False, fill_value=dat['M'][-1])
    R_int = interp1d(dat['M'],dat['r'],bounds_error=False, fill_value=dat['r'][0])

    Rstar = dat['r'][-1]  # Star's radius
    Mstar = dat['M'][-1]  # Star's mass

    # =============================
    # Build SPH star using healpix
    # =============================

    P_mass = Mstar/N_p
    print '\nParticle mass [solar]', P_mass/Mo
    M_shell_min = 12*P_mass
    print 'Lowest shell mass [solar]', M_shell_min/Mo

    r_min = R_int(M_shell_min)

    #Start outside the core

    global r_low

    r_low = r_min

    # ============================================================
    # Obtain positions and masses of SPH particles matching rho(r)

    xpos,ypos,zpos,mp = sph.getSPHParticles(r_low,P_mass,M_int,rho_int,u_int,Rstar,rotshell,gaussRad,Nsigma,dr_sigma,debug=False,writeout_file = f)

    # ====================================================
    # Remove gas particles to be replaced by point mass
    # ====================================================
    if replace_core:

        Mc, N_bdry, xpos, ypos, zpos, mp = ics.remove_gasparticles(xpos,ypos,zpos,mp,R_core)

    # =============================
    # Get SPH particle's properties
    # =============================

    pos = np.zeros((len(xpos),3))
    pos[:,0] = xpos
    pos[:,1] = ypos
    pos[:,2] = zpos

    ptype,id_f,m_f,x_f,y_f,z_f,vx_f,vy_f,vz_f,u_f,h_f,rho_f =\
    sph.get_particle_properties(mp,pos,prel,vrel,vrad,vrot,rho_int,u_int)


    # =============================
    # Add Core point mass and/or BH
    # =============================

    if replace_core:

        # Core properties
        pos_c = np.array([0.0,0.0,0.0])
        vel_c = [0,0,0]
        m_c = Mc

        ptype,id_f,m_f,x_f,y_f,z_f,vx_f,vy_f,vz_f,u_f,h_f,rho_f = sph.add_Particle(3,pos_c,vel_c,m_c,ptype,id_f,m_f,x_f,y_f,z_f,vx_f,vy_f,vz_f,u_f,h_f,rho_f)

        print ''
        print 'Core replaced by a Bulge particle \nsurrounded by a shell with %g SPH particles\n' %(N_bdry)

else:
    ptype = []
    id_f = []
    m_f = []
    x_f = []
    y_f = []
    z_f = []
    vx_f = []
    vy_f = []
    vz_f = []
    u_f = []
    h_f = []
    rho_f = []

if addOneBH:

    ptype,id_f,m_f,x_f,y_f,z_f,vx_f,vy_f,vz_f,u_f,h_f,rho_f = sph.add_Particle(5,posBH,velBH,mBH,ptype,id_f,m_f,x_f,y_f,z_f,vx_f,vy_f,vz_f,u_f,h_f,rho_f)

if addTwoBH:

    ptype,id_f,m_f,x_f,y_f,z_f,vx_f,vy_f,vz_f,u_f,h_f,rho_f = sph.add_Particle(5,posBH1,velBH1,mBH1,ptype,id_f,m_f,x_f,y_f,z_f,vx_f,vy_f,vz_f,u_f,h_f,rho_f)
    
    if starpart == False:
        ptype,id_f,m_f,x_f,y_f,z_f,vx_f,vy_f,vz_f,u_f,h_f,rho_f = sph.add_Particle(5,posBH2,velBH2,mBH2,ptype,id_f,m_f,x_f,y_f,z_f,vx_f,vy_f,vz_f,u_f,h_f,rho_f)
    
    elif starpart == True:
        ptype,id_f,m_f,x_f,y_f,z_f,vx_f,vy_f,vz_f,u_f,h_f,rho_f = sph.add_Particle(4,posBH2,velBH2,mBH2,ptype,id_f,m_f,x_f,y_f,z_f,vx_f,vy_f,vz_f,u_f,h_f,rho_f)

# =============================
# Save data into an hdf5
# =============================

data = Table([ptype,id_f,m_f,x_f,y_f,z_f,vx_f,vy_f,vz_f,u_f,h_f,rho_f]\
             , names=('type','id', 'm', 'x', 'y', 'z', 'vx', 'vy' , 'vz', 'u', 'hsml', 'rho'))

print ''
print '================================================================'
print 'Creating initial conditions with roughly '+str(N_p)+' particles'
print 'in file '+Filename
print '================================================================'

sph.make_hdf5_from_table(data,Filename,scale_to_units,DistUnit,MassUnit,f)


# ====================================================================
## data is already scaled to solar units or whatever units intended
# ====================================================================

data['r'] = np.sqrt(data['x']**2 + data['y']**2 + data['z']**2)

if scale_to_units:
    print '\n-------------------------------------------'
    print 'Scaling distances by ', round(DistUnit,4), ' cm'
    print 'Scaling masses by ', round(MassUnit,4), ' g\n'

else:
    print '\n-------------------------------------------'
    print 'All final data will be in CGS\n'

print''
print'Number of SPH particles in the star'
print len(id_f)

print''
print 'Total number of particles (all types)'
print len(data['id'])

print ''
print '================================='
print 'Done Creating Initial Conditions'
print 'in file '+Filename
print '================================='

print '\n Creating the ICs took ',round(time.time()- start,4),' seconds \n (' ,round(time.time()- start,4)/60., ' minutes)'

direc.write_out(f,Filename,N_p, len(id_f), data['id'], addPoly,addNestedPoly,addOneBH,addTwoBH,scale_to_units)

f.close()

'''
################################################################################


                               Imported Modules


################################################################################


NestedPolyStar: Used to obtain polytropic stars
SphericalSPHDist: Used to build spherical particle distributions
dfunc: Used to create directories and output files after creating initial conditions
BinaryBHOrbit: Used for binary initial conditions


################################################################################


                               Module Options


################################################################################

addPoly (True/False): If True a star with a single given polytropic index will be created.
addNestedPoly (True/False): If True a star with a two given polytropic index will be created.
addOneBH (True/False): If True a single BH will be created.
addTwoBH (True/False): If True a binary BH will be created.
set_coord (True/False): If True the coordinates of all particles will be defined by an IC_Coords file created by the 3body notebook.
starpart (True/False): If True one of the BHs in the binary BH will be a star particle (particle #4) instead of a sink particle (particle #5). 
This may be needed because the angular momentum tracker of accreted particles in Aldo's modified version of Gadget2 can only track the angular momentum accreted onto
a single sink particle.
scale_to_units (True/False): If False units of HDF5 file are in CGS, if True units are as defined.

################################################################################


                               Module Variables


################################################################################

----------------------------
           Units   
----------------------------  

 ** Used to make polytrope **

DistUnit: The distance unit of the SPH simulation, usually a solar radius. [cgs]
MassUnit: The mass unit of the SPH simulations usually a solar mass. [cgs]
TimeUnit: Timestep of simulation, of solar units it's the dynamical time of the sun. [cgs]
DensUnit: The density unit of the SPH simulation. [cgs]
VelUnit: The velocity unit of the SPH simulation. [cgs]
E_perMassUnit: The energy unit per unit mass of the SPH simulation. [cgs]
P_Unit: The pressure unit of the SPH simulation. [cgs]


----------------------------
    Simulation Parameters   
----------------------------  

N_p: Number of SPH particles, simulation usually ends up with a bit less than this value. Also, sometimes if you put in a number that's not a multiple
of ten you may get a weird error (ie. N_p = 5000 gives an error).

Npstring: String of N_p
N_k: Number of thousands of particles in the simulation
N_p_code: Number of particles in the simulation in units of 'k' (ie. 10,000 particles is 10k)

----------------------------
    Stellar Properties 
----------------------------  

Mstar: Star Mass [cgs]
Rstar: Star Radius [cgs]

Sx: x position of star [cgs]
Sy: y position of star [cgs]
Sz: z position of star [cgs]

Svx: x velocity of star [cgs]
Svy: y velocity of star [cgs]
Svz: z velocity of star [cgs]

----------------------------
    Single Polytrope Index 
----------------------------   
n: Polytropic index

----------------------------
    Nested Polytrope 
----------------------------   
n1: Inner Polytropic index for nested polytropic star
n2: Outer Polytropic index for nested polytropic star

----------------------------
    Single BH simulations 
----------------------------    
mBH: Mass of BH [cgs]

BHx: x position of BH [cgs]
BHy: y position of BH [cgs]
BHz: z position of BH [cgs]

BHvx: x velocity of BH [cgs]
BHvy: y velocity of BH [cgs]
BHvz: z velocity of BH [cgs]

----------------------------
    Binary BH simulations 
----------------------------   
mBH1: Mass of BH1 [cgs]
mBH2: Mass of BH2 [cgs]

BH1x: x position of BH1 [cgs]
BH1y: y position of BH1 [cgs]
BH1z: z position of BH1 [cgs]

BH1vx: x velocity of BH1 [cgs]
BH1vy: y velocity of BH1 [cgs]
BH1vz: z velocity of BH1 [cgs]

B2Hx: x position of BH2 [cgs]
B2Hy: y position of BH2 [cgs]
B2Hz: z position of BH2 [cgs]

BH2vx: x velocity of BH2 [cgs]
BH2vy: y velocity of BH2 [cgs]
BH2vz: z velocity of BH2 [cgs]

----------------------------
    Star-Binary Orbit 
----------------------------  

e: The eccentricity of the BBH orbit
P: The period of the BBH [cgs]
sep_at_peri: The separation of the binary when they are closest [cgs]

i: Inclination of the star orbiting the BBH, measured from the BBH orbital plane?
e_orb: Eccentricity of stellar orbit around BBH, usually set to parabolic (e = 1)
pointto: Which BH in the binary the star will be 'pointed' towards

roffset: Sets the initial distance from BBH, overwrites set values for star velocity (Sx,Sy,Sz) [cgs]
v_inf: Sets the initial velocity of the star, overwrites set values for star velocity (Svx,Svy,Svz) [cgs]
rperi: Sets the closest approach to BBH [cgs]

x_orbit: x position of star based on roffset [cgs]
y_orbit: y position of star based on roffset [cgs]
z_orbit: z position of star based on roffset [cgs]

vx_orbit: x velocity of star based on v_inf [cgs]
vy_orbit: y velocity of star based on v_inf [cgs]
vz_orbit: z velocity of star based on v_inf [cgs]
'''
