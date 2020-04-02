import numpy as np
import time
import os
import BinaryBHOrbit as bhb
from distutils.dir_util import copy_tree
from fractions import Fraction

class Create_Directory():

    global Ro, Mo, G, AU, sec_in_day, sec_in_hrs
    Ro=6.96e10
    Mo=1.99e33
    G=6.6726e-08
    AU = 1.49597870e+13
    sec_in_day = 60*60*24
    sec_in_hrs = 60*60

    def __init__(self, N_p_code, DistUnit, MassUnit, VelUnit, Mstar, Rstar, n, n1, n2, mBH, Sx, Sy, Sz, Svx, Svy, Svz, BHx, BHy, BHz, BHvx, BHvy, BHvz,
        mBH1, mBH2, BH1x, BH2x, BH1vx, BH1vy, BH2vx, BH2vy, e, P, sep_at_peri, roffset, rperi, e_orb, i, x_orbit, y_orbit, z_orbit, vx_orbit, 
        vy_orbit, vz_orbit, addPoly, addNestedPoly, addOneBH, addTwoBH):
        
        #Simulation Variables
        self.N_p_code = N_p_code
        self.DistUnit = DistUnit
        self.MassUnit = MassUnit
        self.VelUnit = VelUnit
        
        #Polytrope Parameters
        self.Mstar = Mstar 
        self.Rstar = Rstar 
        self.n = n 

        #Nested Polytrope Parameters
        self.n1 = n1 
        self.n2 = n2 

        #One BH Parameters
        self.mBH = mBH
        
        self.Sx = Sx
        self.Sy = Sy
        self.Sz = Sz

        self.Svx = Svx
        self.Svy = Svy
        self.Svz = Svz
        
        self.BHx = BHx
        self.BHy = BHy
        self.BHz = BHz

        self.BHvx = BHvx
        self.BHvy = BHvy
        self.BHvz = BHvz

        #Two BH Parameters
        self.mBH1 = mBH1 
        self.mBH2 = mBH2
        
        self.BH1x = BH1x
        self.BH2x = BH2x
        
        self.BH1vx = BH1vx
        self.BH1vy = BH1vy
        
        self.BH2vx = BH2vx
        self.BH2vy = BH2vy 
        
        self.e = e 
        self.P = P
        
        #Stellar Orbit Parameters
        self.sep_at_peri = sep_at_peri 
        self.roffset = roffset 
        self.rperi = rperi 
        self.e_orb = e_orb 
        self.i = i 
        
        self.x_orbit = x_orbit
        self.y_orbit = y_orbit
        self.z_orbit = z_orbit
        self.vx_orbit = vx_orbit
        self.vy_orbit = vy_orbit
        self.vz_orbit = vz_orbit

        #String Variables

        if addPoly or addNestedPoly:
            self.Sx_str = str(round(self.Sx/Ro,2)) + ' [Ro]'
            self.Sy_str = str(round(self.Sy/Ro,2)) + ' [Ro]'
            self.Sz_str = str(round(self.Sz/Ro,2)) + ' [Ro]'

            self.Svx_str = str(round(self.Svx/(1.0e5),2)) + ' [kms]'
            self.Svy_str = str(round(self.Svy/(1.0e5),2)) + ' [kms]'
            self.Svz_str = str(round(self.Svz/(1.0e5),2)) + ' [kms]'

        if addOneBH:
            self.BHx_str = str(round(self.BHx/Ro,2)) + ' [Ro]'
            self.BHy_str = str(round(self.BHy/Ro,2)) + ' [Ro]'
            self.BHz_str = str(round(self.BHz/Ro,2)) + ' [Ro]'

            self.BHvx_str = str(round(self.BHvx/(1.0e5),2)) + ' [kms]'
            self.BHvy_str = str(round(self.BHvy/(1.0e5),2)) + ' [kms]'
            self.BHvz_str = str(round(self.BHvz/(1.0e5),2)) + ' [kms]'

        if addTwoBH:
            #BH Binary Properties
            self.BH1x_str = str(round(self.BH1x/Ro,2)) + ' [Ro]'
            self.BH2x_str = str(round(self.BH2x/Ro,2)) + ' [Ro]'

            self.BH1vx_str = str(round(self.BH1vx/(1.0e5),2)) + ' [kms]'
            self.BH1vy_str = str(round(self.BH1vy/(1.0e5),2)) + ' [kms]'

            self.BH2vx_str = str(round(self.BH2vx/(1.0e5),2)) + ' [kms]'
            self.BH2vy_str = str(round(self.BH2vy/(1.0e5),2)) + ' [kms]'

            self.BH1v_str = str(round(np.sqrt(self.BH1vx**2 + self.BH1vy**2)/(1.0e5),2)) + ' [kms]'
            self.BH2v_str = str(round(np.sqrt(self.BH2vx**2 + self.BH2vy**2)/(1.0e5),2)) + ' [kms]'

            self.e_str = str(self.e) + ' e'
            self.T_str = str(round(self.P/sec_in_day,2)) + ' [days]'
        
            if addPoly or addNestedPoly:    
                #Stellar Orbit Properties
                self.x_orbit_str = str(round(self.x_orbit/Ro,2)) + ' [Ro]'
                self.y_orbit_str = str(round(self.y_orbit/Ro,2)) + ' [Ro]'
                self.z_orbit_str = str(round(self.z_orbit/Ro,2)) + ' [Ro]'

                self.r_orbit_str = str(round(np.sqrt(self.x_orbit**2 + self.y_orbit**2 + self.z_orbit**2)/Ro,2)) + ' [Ro]'

                self.vx_orbit_str = str(round(self.vx_orbit/(1.0e5),2)) + ' [kms]'
                self.vy_orbit_str = str(round(self.vy_orbit/(1.0e5),2)) + ' [kms]'
                self.vz_orbit_str = str(round(self.vz_orbit/(1.0e5),2)) + ' [kms]'

                self.v_orbit_str = str(round(np.sqrt(self.vx_orbit**2 + self.vy_orbit**2 + self.vz_orbit**2)/(1.0e5),2)) + ' [kms]'

                self.i_str = str(Fraction(round(self.i/np.pi,2)).limit_denominator()) + ' pi radians'
        
        print ("Initial variables defined...\n")
        return None

    def write_out(self, filename, HDF5_file, N_p, N_env, data_id, addPoly = False, addNestedPoly = False, addOneBH = False, addTwoBH = False, scale_to_units = False):
        filename.write(
            '\nNumber of SPH particles in the star\n' +
            str(N_env) + '\n'

            '\nTotal number of particles (all types)\n' +
            str(len(data_id)) + '\n'
 
            '\n' +
            '================================================================\n' +
            'Creating initial conditions with roughly '+ self.N_p_code +' particles\n' +
            'in file '+ HDF5_file + '\n' +
            '================================================================\n'
            )

        if scale_to_units == True:
            filename.write(
            '\n-------------------------------------------\n' +
            'Scaling distances by ' + str(round(self.DistUnit,4)) + ' cm\n' +
            'Scaling masses by ' + str(round(self.MassUnit,4)) + ' g\n'
            )
        elif scale_to_units == False:
            filename.write(
            '\n-------------------------------------------\n' +
            'All final data will be in CGS\n'
            )

        if addPoly or addNestedPoly:
            filename.write(
            '\n' +
            '================================================================\n' +
            'Stellar Properties\n' + 
            'Sx: ' + self.Sx_str + '\n' + 
            'Sy: ' + self.Sy_str + '\n' + 
            'Sz: ' + self.Sz_str + '\n' + 
            '\n'
            'Svx: ' + self.Svx_str + '\n' + 
            'Svy: ' + self.Svy_str + '\n' + 
            'Svz: ' + self.Svz_str + '\n' + 
            '================================================================\n\n'
            )

        if addOneBH:
            filename.write(
            '\n' +
            '================================================================\n' +
            'Black Hole Properties\n' +             
            'BHx: ' + self.BHx_str + '\n' + 
            'BHy: ' + self.BHy_str + '\n' + 
            'BHz: ' + self.BHz_str + '\n' + 
            '\n'
            'BHvx: ' + self.BHvx_str + '\n' + 
            'BHvy: ' + self.BHvy_str + '\n' + 
            'BHvz: ' + self.BHvz_str + '\n' + 
            '================================================================\n\n'
            )        

        if addTwoBH:
            filename.write(
            '\n' +
            '================================================================\n' +
            'Binary Orbit Properties\n' + 
            'BH1x: ' + self.BH1x_str + '\n' + 
            'BH2x: ' + self.BH2x_str + '\n' + 
            '\n'
            'BH1vx: ' + self.BH1vx_str + '\n' + 
            'BH1vy: ' + self.BH1vy_str + '\n' + 
            '\n'
            'BH2vx: ' + self.BH2vx_str + '\n' + 
            'BH2vy: ' + self.BH2vy_str + '\n' + 
            '\n'
            'BHv1: ' + self.BH1v_str + '\n' + 
            'BHv2: ' + self.BH2v_str + '\n' +
            '\n' +
            'e: ' + self.e_str + '\n' +
            'T: ' + self.T_str + '\n' +
            '================================================================\n\n'
            )
    
            if addPoly or addNestedPoly:
                filename.write(
                '\n' +
                '================================================================\n' +
                'Stellar Orbit Properties\n' + 
                'x_orbit: ' + self.x_orbit_str + '\n' + 
                'y_orbit: ' + self.y_orbit_str + '\n'
                'z_orbit: ' + self.z_orbit_str + '\n'
                '\n'
                'r_orbit: ' + self.r_orbit_str + '\n'
                '\n'
                'vx_orbit: ' + self.vx_orbit_str + '\n'
                'vy_orbit: ' + self.vy_orbit_str + '\n'
                'vz_orbit: ' + self.vz_orbit_str + '\n'
                '\n'
                'v_orbit: ' + self.v_orbit_str + '\n'
                '\n'
                'inclination: ' + self.i_str + '\n'
                '================================================================\n'
                )

    def create_path(self,new_path,sim = False):
        if not os.path.exists(new_path):
            if sim == False:
                os.makedirs(new_path)

            elif sim == True:
                run_path = new_path + '/run1'
                log_path = new_path + '/Sim_Log'
                os.makedirs(run_path)

                # Copies master LaTeX file into each simulation for logging results
                fromDirectory1 = 'Sim_Log'
                toDirectory1 = log_path
                copy_tree(fromDirectory1, toDirectory1)

                fromDirectory2 = 'run1'
                toDirectory2 = run_path
                copy_tree(fromDirectory2, toDirectory2)

    def create_IC(self,addPoly = False, addNestedPoly = False, addOneBH = False, addTwoBH = False):
        
        if addPoly:

            out = 'P'
            DATprofile = str(int(self.Mstar/Mo)) + 'M' + str(int(self.Rstar/Ro)) +'R'
            polyindex = 'n' + str(self.n)

            #This is the path of the density profile
            DATfilename = '../DAT/' + DATprofile + '_' + polyindex + '.dat'
            
        if addNestedPoly:

            out = 'NP'

            DATprofile = str(int(self.Mstar/Mo)) + 'M_' + str(int(self.Rstar/Ro)) +'R'
            polyindex = 'nc' + str(int(self.n1)) + 'ne' + str(self.n2)

            #This is the path of the density profile
            DATfilename = '../DAT/' + DATprofile + '_' + polyindex + '.dat'

        if addOneBH:

            #Converts quantities from cgs to solar units
            BHy_code = int(self.BHy/self.DistUnit)
            Svx_code = int(self.Svx/1e5)
            mBH_code = str(int(self.mBH / self.MassUnit)) + 'M'

            if addPoly or addNestedPoly:
                DATprofile = DATprofile + '-' + mBH_code
            else:
                DATprofile = mBH_code

            params =  str(BHy_code) + 'y_' + str(Svx_code) + 'vx'

        if addTwoBH:

            if addPoly:
                r_code = str(int(self.roffset/Ro)) + 'r'
                rp_code = str(int(self.rperi/Ro)) + 'rp'

            #Converts values to str in CGS
            e_str = str(self.e) + 'e'
            T_str = str(round(self.P,2))

            #Converts from CGS to code units
            sep_code = str(int(round(self.sep_at_peri / self.DistUnit))) + 'a'
            mBH1_code = str(int(self.mBH1 / self.MassUnit)) + 'M'
            mBH2_code = str(int(self.mBH2 / self.MassUnit)) + 'M'

            if addPoly or addNestedPoly:
                DATprofile = DATprofile + '-' + mBH1_code + mBH2_code + '_' + sep_code + e_str
                params =  r_code + rp_code

            else:
                DATprofile = mBH1_code + mBH2_code + '_' + sep_code + e_str
                params = DATprofile

        if not(addPoly) and not(addNestedPoly) and addTwoBH:
            out = 'BHb'
            DATfilename = None

        if not(addOneBH) and not(addTwoBH):
            params = DATprofile

        if addNestedPoly:

            path = '../HDF5/' + polyindex + '/' + N_p_code + '/' + DATprofile + '/' + coretype + '/'

            #Creates the directory where to save the out files
            out_path = '/Users/martinlopezjr/Dropbox/out/' + out + '/' + polyindex + '/' + self.N_p_code + '/' + DATprofile + '/' + coretype + '/' +  params

            #This is the path of the SPH particles
            Filename = path + params + '.hdf5'
            valname =  path + params + '_values.txt'

        elif addPoly:
            path = '../HDF5/' + polyindex + '/' + self.N_p_code + '/' + DATprofile + '/'

            #Creates the directory where to save the out files
            out_path = '/Users/martinlopezjr/Dropbox/out/' + out + '/' + polyindex + '/' + self.N_p_code + '/' + DATprofile + '/' +  params

            #This is the path of the SPH particles
            Filename =  path + params + '.hdf5'
            valname =  path + params + '_values.txt'

        else:
            path = '../HDF5/' + out + '/' + DATprofile + '/'

            #Creates the directory where to save the out files
            out_path = '/Users/martinlopezjr/Dropbox/out/' + out + '/' + DATprofile

            #This is the path of the SPH particles
            Filename =  path + params + '.hdf5'
            valname =  path + params + '_values.txt'

        #Creates the directory where to save the IC
        self.create_path(path)
        self.create_path(out_path, sim = True)

        return Filename, valname, DATfilename
