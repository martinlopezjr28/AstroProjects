########################################
#                HEADER
########################################

import numpy as np
import glob as glb
import readsnapgadget as rs
import h5py as h

from astropy.io import ascii
from astropy.table import Table
from scipy import stats
from scipy import integrate


class Constants(object):

    # Physical Constants
    G = 6.674e-8           # Gravitational Constant [g^-1 s^-2 cm^3]
    c = 2.99792458e10      # Speed of Light [cm s^-1]
    mp = 1.6726219e-24     # Mass of Proton [g]
    me = 9.10938356e-28    # Mass of Electron [g]
    kB = 1.380658e-16      # Boltzmann Constant

    # Astronomical Constants
    AU = 1.496e13          # Astronomical Unit [cm]
    pc = 3.085677581e18    # Parsec [cm]

    # Solar Constants
    Mo = 1.989e33                        # Solar Mass [g]
    Ro = 6.955e10                         # Solar Radius [cm]
    Lo = 3.9e33                          # Solar Luminosity [erg s^-1]
    To = 5.78e3                          # Solar Temperature [K]
    Go = 1.0                             # Solar G constant
    # Solar Escape Velocity [cm s^-1]
    Vo = np.sqrt(2 * G * Mo / Ro)
    # Solar Average Density [g cm^-3]
    Rhoo = Mo / ((4. / 3.) * np.pi * Ro**3)

    # Conversion Factors
    sec_per_year = 3.1536e7
    sec_per_day = 60. * 60. * 24
    sec_per_hrs = 60. * 60.
    sec_per_min = 60.
    sec_per_tdyn = np.sqrt(Ro**3 / (G * Mo))

    print "Constants defined.\n"


class PhysicalQuantities(Constants):
    """
        Extracts physical quantites from SPH simulations using HDF5 outputs.
        All quantities are in solar units unless specified otherwise.

        Attributes
        ----------
        path : str
            contains the path to either the simulation folder or a single
            simulation snapshot
        longsim : True/False
            if True, snapshot file names are formatted as 'snapshot_0001'
            if false, snapshot file names are formatted as 'snapshot_001'
        id_BH1 : int
            the id of the "first" BH in the simulation
        id_BH2 : int
            the id of the "second" BH in the simulation
        snapfactor : int, optional
            sets how often snapshots are read, every 'snapfactor'
            snapshots will be read (default is 1)

        Methods
        ----------
        get_init(num)
            Gets the data from simulations that all other methods will use
            for snapshot 'num'
        get_pos(data_pos_gas, data_pos_BH, mgas, mBH1, mBH2, mBHtotal,
                bool_BH1, bool_BH2)
            Gets the absolute and relative position and distance quantities
            for a single simulation snapshot
        get_vel(data_vel_gas, data_vel_BH, mgas, mBH1, mBH2, mBHtotal,
                bool_BH1, bool_BH2)
            Gets the absolute and relative velocity and distance quantities
            for a single simulation snapshot
        get_E(data_pos_gas, data_vel_gas, data_pos_BH, data_vel_BH, mgas, mBH1,
              mBH2, mBHtotal, bool_BH1, bool_BH2)
            gets the kinetic, potential, and total energy of a single
            simulation snapshot along with the booleans labelling bound
            particles relative to each BH and the BBH CM for a single
            simulation snapshot
        get_mdot(data_pos_gas, data_vel_gas, data_pos_BH, data_vel_BH, mgas, mBH1,
                 mBH2, mBHtotal, bool_BH1, bool_BH2, mdot_radius)
            Gets the accretion rates for each BH and BBH CM defined
            the 'mdot_radius', calculates the number of particles bound,
            within the 'mdot_radius', and bound within the 'mdot_radius'
            seperately for a single simulation snapshot
        get_L(data_pos_gas, data_vel_gas, data_pos_BH, data_vel_BH, mgas, mBH1,
              mBH2, mBHtotal, bool_BH1, bool_BH2)
            Gets the total angular momentum of the system and the angular
            momentum of each particle with respect to each BH for a single
            simulation snapshot
        get_angle(data_pos_gas, data_vel_gas, data_pos_BH, data_vel_BH, mgas,
                  mBH1, mBH2, mBHtotal, bool_BH1, bool_BH2)
            Gets the angle between the angular momentum all gas particles
            with respect to each BH and the orbital angular
            momentum of the BBH
        init_pos(init_file=None)
            Initializes all lists and arrays relevant to the get_pos method
            for all snapshots being evaluated, if an 'init_file' exists
            this method will get needed data from 'init_file' to execute
        init_vel(init_file=None)
            Initializes all lists and arrays relevant to the get_vel method
            for all snapshots being evaluated, if an 'init_file' exists
            this method will get needed data from 'init_file' to execute
        init_E(init_file=None)
            Initializes all lists and arrays relevant to the get_E method
            for all snapshots being evaluated, if an 'init_file' exists
            this method will get needed data from 'init_file' to execute
        init_mdot(mdot_radius, tdyn_per_snap=1, init_file=None)
            Initializes all lists and arrays relevant to the get_mdot method
            for all snapshots being evaluated, if an 'init_file' exists
            this method will get needed data from 'init_file' to execute,
            'mdot_radius' must be passed to get_mdot and 'tdyn_per_snap'
            will be used to correctly calculate the dt between snapshots
        init_L(init_file=None)
            Initializes all lists and arrays relevant to the get_L method
            for all snapshots being evaluated, if an 'init_file' exists
            this method will get needed data from 'init_file' to execute
        init_angle(init_file=None)
            Initializes all lists and arrays relevant to the get_angle method
            for all snapshots being evaluated, if an 'init_file' exists
            this method will get needed data from 'init_file' to execute
        write_hdf5(write_file, list_type, init_file=None, mdot_radius=1,
                   tdyn_per_snap=1)
            Writes an hdf5 file saving all data specific to 'list_type', list_types options currently available are: 'init', 'pos',
            'vel', 'mdot', 'L', 'E', 'angle'. 'mdot_radius' must be specified
            when using list_type='mdot'
        read_hdf5(read_file)
            Reads hdf5 file 'read_file' and returns a list containing all
            datasets stored in 'read_file' and prints out the dataset-index
            mapping
    """

    def __init__(self, path, longsim, id_BH1, id_BH2, snapfactor=1):
        """
        Parameters
        ----------
        path : str
        contains the path to either the simulation folder or a single
        simulation snapshot
        longsim : True/False
            if True, snapshot file names are formatted as 'snapshot_0001'
            if false, snapshot file names are formatted as 'snapshot_001'
        id_BH1 : int
            the id of the "first" BH in the simulation
        id_BH2 : int
            the id of the "second" BH in the simulation
        snapfactor : int, optional
        sets how often snapshots are read, every 'snapfactor'
        snapshots will be read (default is 1)
        """

        # Initializations

        # simulation resolution
        self.res = None

        # list with all of the snapshots
        self.numsnap = []

        # gas data for each snapshot
        self.data_gas_list = []
        self.data_pos_gas_list = []
        self.data_vel_gas_list = []

        # BH data for each snapshot
        self.data_BH_list = []
        self.data_pos_BH_list = []
        self.data_vel_BH_list = []

        # BH IDs
        self.bool_BH1_list = []
        self.bool_BH2_list = []

        # masses
        self.mgas_list = []
        self.mgas_total_list = []

        self.mBH1_list = []
        self.mBH2_list = []
        self.mBHtotal_list = []

        # position quantities
        self.pos_gas_list = []
        self.pos_BH1_list = []
        self.pos_BH2_list = []

        self.pos_gasCM_list = []
        self.pos_BHCM_list = []

        # distance quantities:
        self.pos_gas2BH1_list = []
        self.pos_gas2BH2_list = []
        self.pos_gas2BHCM_list = []

        self.pos_gasCM2BH1_list = []
        self.pos_gasCM2BH2_list = []
        self.pos_gasCM2BHCM_list = []

        self.d_gas2BH1_list = []
        self.d_gas2BH2_list = []
        self.d_gas2BHCM_list = []

        self.d_gasCM2BH1_list = []
        self.d_gasCM2BH2_list = []
        self.d_gasCM2BHCM_list = []

        self.separation_list = []

        # velocity quantities
        self.vel_gas_list = []
        self.vel_BH1_list = []
        self.vel_BH2_list = []

        self.vel_gasCM_list = []
        self.vel_BHCM_list = []

        self.vel_gas_mag_list = []
        self.vel_BH1_mag_list = []
        self.vel_BH2_mag_list = []
        self.vel_BHCM_mag_list = []

        # mdot quantities
        self.N_mdotBH1_list = []
        self.N_mdotBH2_list = []
        self.N_mdotBHCM_list = []

        self.N_mdotBH1_bound_list = []
        self.N_mdotBH2_bound_list = []
        self.N_mdotBHCM_bound_list = []

        self.mdot_tot_BH1_list = []
        self.mdot_tot_BH2_list = []
        self.mdot_tot_BHCM_list = []

        # total angular momentum quantities
        self.L_gas_list = []
        self.L_BH1_list = []
        self.L_BH2_list = []
        self.L_total_list = []

        # relative angular momentum quantities
        self.L_gas2BH1_list = []
        self.L_gas2BH2_list = []
        self.L_gas2BHCM_list = []

        self.L_gasCM2BH1_list = []
        self.L_gasCM2BH2_list = []
        self.L_gasCM2BHCM_list = []

        # particle kinetic energy lists
        self.T_gas2BH1_list = []
        self.T_gas2BH2_list = []
        self.T_gas2BHCM_list = []

        # particle potential energy lists
        self.U_gas2BH1_list = []
        self.U_gas2BH2_list = []
        self.U_gas2BHCM_list = []

        # total particle energy lists
        self.E_gas2BH1_list = []
        self.E_gas2BH2_list = []
        self.E_gas2BHCM_list = []

        # total system energy lists
        self.E_gas2BH1_tot_list = []
        self.E_gas2BH2_tot_list = []
        self.E_gas2BHCM_tot_list = []

        # the boolean arrays for bound particles
        self.bool_gas2BH1_bound_list = []
        self.bool_gas2BH2_bound_list = []
        self.bool_gas2BHCM_bound_list = []

        # the angle of each gas particle to the BH
        self.angle_gas2BH1_list = []
        self.angle_gas2BH2_list = []

        self.angle_gasCM2BH1_list = []
        self.angle_gasCM2BH2_list = []
        self.angle_gasCM2BHCM_list = []

        # eccentricity
        self.e_list = []

        # unique instance definitions
        self.path = path
        self.longsim = longsim
        self.id_BH1 = id_BH1
        self.id_BH2 = id_BH2
        self.snapfactor = snapfactor

        if self.path[-1] == '5':

            # find snapshots in path
            self.snapshots = [self.path]
            self.path = self.path[:-19]

        elif self.path[-1] == '/':

            # find snapshots in path
            self.snapshots = glb.glob(self.path + 'snapshot_*')
            self.snapshots = np.sort(self.snapshots)

        for j, snap in enumerate(self.snapshots):

            # obtain list with snapshot numbers in path
            if self.longsim is True:
                self.numsnap.append(int(self.snapshots[j][-9:-5]))

            else:
                self.numsnap.append(int(self.snapshots[j][-8:-5]))

        # truncates the original numsnap list
        self.numsnap = self.numsnap[::self.snapfactor]

        # changes list to an array
        self.numsnap = np.array(self.numsnap)

    #######################################
    #       Get Methods
    #######################################

    def get_init(self, num):
        """
        Gets the data from simulations that all other methods will use
        for snapshot 'num'

        Parameters
        -------
        num : int
            The number of the snapshot being passed


        Returns
        -------
        data_gas : list
            All gas data for each particle from snapshot
        data_pos_gas : list
            Position gas data for each particle
        data_vel_gas : list
            Velocity gas data for each particle
        data_BH : list
            All black hole data
        data_pos_BH : list
            Position data for each BH
        data_vel_BH : list
            Velocity data for each BH
        bool_BH1 : int
            The index of BH1 when using data_BH
        bool_BH2 : int
            The index of BH2 when using data_BH
        mgas_total : float
            The total mass of gas particles
        mgas : list
            The mass of each gas particle
        mBH1 : float
            The mass of BH1
        mBH2 : float
            The mass of BH2
        mBHtotal : float
            The mass of BH1 + BH2
        """
        self.num = num

        print 'reading snapshot', num

        # reading in gas particle data
        data_gas = rs.readsnap(self.path, snum=num, ptype=0,
                               four_char=self.longsim)
        data_pos_gas = data_gas['p']
        data_vel_gas = data_gas['v']

        # reading in BHB data
        data_BH = rs.readsnap(self.path, snum=num, ptype=5,
                              four_char=self.longsim)
        data_pos_BH = data_BH['p']
        data_vel_BH = data_BH['v']

        res = len(data_gas['id'])

        if np.logical_and(res < 1e4, res > 1e3):
            res = '10k'
        elif np.logical_and(res < 1e5, res > 1e4):
            res = '100k'
        elif np.logical_and(res < 1e6, res > 1e5):
            res = '1000k'

        # Tags the black holes by ID because they move around within the list
        for i, id in enumerate(data_BH['id']):
            if data_BH['id'][0] == self.id_BH1:
                bool_BH1 = 0
                bool_BH2 = 1

            elif data_BH['id'][0] == self.id_BH2:
                bool_BH1 = 1
                bool_BH2 = 0

        mgas = data_gas['m']
        mgas_total = np.sum(mgas)

        # Gets the mass of each black hole
        mBH1 = data_BH['m'][bool_BH1]
        mBH2 = data_BH['m'][bool_BH2]
        mBHtotal = mBH1 + mBH2

        return data_gas, data_pos_gas, data_vel_gas, data_BH, data_pos_BH, data_vel_BH, bool_BH1, bool_BH2, mgas_total, mgas, mBH1, mBH2, mBHtotal

    def get_pos(self, data_pos_gas, data_pos_BH, mgas, mBH1, mBH2, mBHtotal, bool_BH1, bool_BH2):

        """
        Gets the data from simulations that all other methods will use
        for snapshot 'num'

        Parameters
        -------
        data_pos_gas : list
            Position gas data for each particle
        data_pos_BH : list
            Position data for each BH
        mgas : list
            The mass of each gas particle
        mBH1 : float
            The mass of BH1
        mBH2 : float
            The mass of BH2
        mBHtotal : float
            The mass of BH1 + BH2
        bool_BH1 : int
            The index of BH1 when using data_BH
        bool_BH2 : int
            The index of BH2 when using data_BH

        Returns
        -------
        pos_gas : array
            An array containing the absolute XYZ position of each gas
            particle
        pos_BH1 : array
            An array containing the absolute XYZ position of BH1
            particle
        pos_BH2 : array
            An array containing the absolute XYZ position of BH2
            particle
        pos_gasCM : array
            An array containing the absolute XYZ position of the CM of
            all gas particles
        pos_BHCM : array
            An array containing the absolute XYZ position of the BBH CM
        pos_gas2BH1 : array
            An array containing the relative XYZ position of all gas
            particles to BH1
        pos_gas2BH2 : array
            An array containing the relative XYZ position of all gas
            particles to BH2
        pos_gas2BHCM : array
            An array containing the relative XYZ position of all gas
            particles to the BBH CM
        pos_gasCM2BH1 : array
            An array containing the relative XYZ position of the gas CM
            to BH1
        pos_gasCM2BH2 : array
            An array containing the relative XYZ position of the gas CM
            to BH2
        pos_gasCM2BHCM : array
            An array containing the relative XYZ position of the gas CM
            to the BBH CM
        d_gas2BH1 : float
            The distance all gas particles to BH1
        d_gas2BH2 : array
            The distance all gas particles to BH2
        d_gas2BHCM : array
            The distance all gas particles to the BBH CM
        d_gasCM2BH1 : array
            The distance the gas CM to BH1
        d_gasCM2BH2 : array
            The distance the gas CM to BH2
        d_gasCM2BHCM : array
            The distance the gas CM to the BBH CM
        separation : array
            The distance between BH1 and BH2
        """

        self.data_pos_gas = data_pos_gas
        self.data_pos_BH = data_pos_BH

        self.bool_BH1 = bool_BH1
        self.bool_BH2 = bool_BH2

        self.mgas = mgas
        self.mBH1 = mBH1
        self.mBH2 = mBH2
        self.mBHtotal = mBHtotal

        gas_x = self.data_pos_gas[:, 0]
        gas_y = self.data_pos_gas[:, 1]
        gas_z = self.data_pos_gas[:, 2]

        # Gets the center of mass of each position coordinate
        gasCM_x = np.sum(self.mgas * gas_x) / np.sum(self.mgas)
        gasCM_y = np.sum(self.mgas * gas_y) / np.sum(self.mgas)
        gasCM_z = np.sum(self.mgas * gas_z) / np.sum(self.mgas)

        # Puts position and CM values into N x 3 arrays
        pos_gas = np.array([gas_x, gas_y, gas_z])
        pos_gasCM = np.array([gasCM_x, gasCM_y, gasCM_z])

        # import pdb

        # pdb.set_trace()

        # Repeats above steps for each BH
        BH1_x = self.data_pos_BH[self.bool_BH1][0]
        BH1_y = self.data_pos_BH[self.bool_BH1][1]
        BH1_z = self.data_pos_BH[self.bool_BH1][2]

        pos_BH1 = np.array([BH1_x, BH1_y, BH1_z])

        BH2_x = self.data_pos_BH[self.bool_BH2][0]
        BH2_y = self.data_pos_BH[self.bool_BH2][1]
        BH2_z = self.data_pos_BH[self.bool_BH2][2]

        pos_BH2 = np.array([BH2_x, BH2_y, BH2_z])

        # Gets the center of mass of each position coordinate
        BHCM_x = (mBH1 * BH1_x + mBH2 * BH2_x) / (mBH1 + mBH2)
        BHCM_y = (mBH1 * BH1_y + mBH2 * BH2_y) / (mBH1 + mBH2)
        BHCM_z = (mBH1 * BH1_z + mBH2 * BH2_z) / (mBH1 + mBH2)

        pos_BHCM = np.array([BHCM_x, BHCM_y, BHCM_z])

        # Gets the relative positions of the system

        # calculates the distance between all gas particles and BH1
        gas2BH1_x = gas_x - BH1_x
        gas2BH1_y = gas_y - BH1_y
        gas2BH1_z = gas_z - BH1_z

        # calculates the distance between all gas particles and BH2
        gas2BH2_x = gas_x - BH2_x
        gas2BH2_y = gas_y - BH2_y
        gas2BH2_z = gas_z - BH2_z

        # calculates the distance between all gas particles and the BBH CM
        gas2BHCM_x = gas_x - BHCM_x
        gas2BHCM_y = gas_y - BHCM_y
        gas2BHCM_z = gas_z - BHCM_z

        # calculates the distance between the gas CM and BH1
        gasCM2BH1_x = gasCM_x - BH1_x
        gasCM2BH1_y = gasCM_y - BH1_y
        gasCM2BH1_z = gasCM_z - BH1_z

        # calculates the distance between the gas CM and BH2
        gasCM2BH2_x = gasCM_x - BH2_x
        gasCM2BH2_y = gasCM_y - BH2_y
        gasCM2BH2_z = gasCM_z - BH2_z

        # calculates the distance between the gas CM and the BBH CM
        gasCM2BHCM_x = gasCM_x - BHCM_x
        gasCM2BHCM_y = gasCM_y - BHCM_y
        gasCM2BHCM_z = gasCM_z - BHCM_z

        pos_gas2BH1 = np.array([gas2BH1_x, gas2BH1_y, gas2BH1_z])
        pos_gas2BH2 = np.array([gas2BH2_x, gas2BH2_y, gas2BH2_z])
        pos_gas2BHCM = np.array([gas2BHCM_x, gas2BHCM_y, gas2BHCM_z])

        pos_gasCM2BH1 = np.array([gasCM2BH1_x, gasCM2BH1_y, gasCM2BH1_z])
        pos_gasCM2BH2 = np.array([gasCM2BH2_x, gasCM2BH2_y, gasCM2BH2_z])
        pos_gasCM2BHCM = np.array([gasCM2BHCM_x, gasCM2BHCM_y, gasCM2BHCM_z])

        d_gas2BH1 = np.sqrt(gas2BH1_x**2 + gas2BH1_y**2 + gas2BH1_z**2)
        d_gas2BH2 = np.sqrt(gas2BH2_x**2 + gas2BH2_y**2 + gas2BH2_z**2)
        d_gas2BHCM = np.sqrt(gas2BHCM_x**2 + gas2BHCM_y**2 + gas2BHCM_z**2)

        d_gasCM2BH1 = np.sqrt(gasCM2BH1_x**2 + gasCM2BH1_y**2 + gasCM2BH1_z**2)
        d_gasCM2BH2 = np.sqrt(gasCM2BH2_x**2 + gasCM2BH2_y**2 + gasCM2BH2_z**2)
        d_gasCM2BHCM = np.sqrt(gasCM2BHCM_x**2 + gasCM2BHCM_y**2 +
                               gasCM2BHCM_z**2)

        # Calculates the separation between black holes
        separation = np.sqrt((BH1_x - BH2_x)**2 +
                             (BH1_y - BH2_y)**2 +
                             (BH1_z - BH2_z)**2)

        return pos_gas, pos_BH1, pos_BH2, pos_gasCM, pos_BHCM, pos_gas2BH1,\
        pos_gas2BH2, pos_gas2BHCM,pos_gasCM2BH1, pos_gasCM2BH2,\
        pos_gasCM2BHCM,d_gas2BH1, d_gas2BH2, d_gas2BHCM, d_gasCM2BH1,\
        d_gasCM2BH2, d_gasCM2BHCM, separation

    def get_vel(self, data_vel_gas, data_vel_BH, mgas, mBH1, mBH2, mBHtotal, bool_BH1, bool_BH2):

        # Gets the velocity arrays of each coordinate to
        # calculate the velocity of center of mass

        self.data_vel_gas = data_vel_gas
        self.data_vel_BH = data_vel_BH

        self.bool_BH1 = bool_BH1
        self.bool_BH2 = bool_BH2

        self.mgas = mgas
        self.mBH1 = mBH1
        self.mBH2 = mBH2
        self.mBHtotal = mBHtotal

        # breaks down the 2-d gas position array into 3 1-d position arrays
        gas_vx = self.data_vel_gas[:, 0]
        gas_vy = self.data_vel_gas[:, 1]
        gas_vz = self.data_vel_gas[:, 2]

        # Gets the center of mass of each position coordinate
        gasCM_vx = np.sum(self.mgas * gas_vx) / np.sum(self.mgas)
        gasCM_vy = np.sum(self.mgas * gas_vy) / np.sum(self.mgas)
        gasCM_vz = np.sum(self.mgas * gas_vz) / np.sum(self.mgas)

        vel_gas_mag = np.sqrt(gas_vx**2 + gas_vy**2 + gas_vz**2)
        gasCM_vmag = np.sqrt(gasCM_vx**2 + gasCM_vy**2 + gasCM_vz**2)

        # breaks down the 2-d BH position array into 3 1-d position arrays
        BH1_vx = self.data_vel_BH[bool_BH1][0]
        BH1_vy = self.data_vel_BH[bool_BH1][1]
        BH1_vz = self.data_vel_BH[bool_BH1][2]

        BH2_vx = self.data_vel_BH[bool_BH2][0]
        BH2_vy = self.data_vel_BH[bool_BH2][1]
        BH2_vz = self.data_vel_BH[bool_BH2][2]

        # Gets the center of mass of each position coordinate
        BHCM_vx = (self.mBH1 * BH1_vx + self.mBH2 * BH2_vx) / self.mBHtotal
        BHCM_vy = (self.mBH1 * BH1_vy + self.mBH2 * BH2_vy) / self.mBHtotal
        BHCM_vz = (self.mBH1 * BH1_vz + self.mBH2 * BH2_vz) / self.mBHtotal

        vel_BH1_mag = np.sqrt(BH1_vx**2 + BH1_vy**2 + BH1_vz**2)
        vel_BH2_mag = np.sqrt(BH2_vx**2 + BH2_vy**2 + BH2_vz**2)
        vel_BHCM_mag = np.sqrt(BHCM_vx**2 + BHCM_vy**2 + BHCM_vz**2)


        # relative velocities
        gas2BH1_vx = gas_vx - BH1_vx
        gas2BH1_vy = gas_vy - BH1_vy
        gas2BH1_vz = gas_vz - BH1_vz

        gas2BH2_vx = gas_vx - BH2_vx
        gas2BH2_vy = gas_vy - BH2_vy
        gas2BH2_vz = gas_vz - BH2_vz

        gas2BHCM_vx = gas_vx - BHCM_vx
        gas2BHCM_vy = gas_vy - BHCM_vy
        gas2BHCM_vz = gas_vz - BHCM_vz

        gasCM2BH1_vx = gasCM_vx - BH1_vx
        gasCM2BH1_vy = gasCM_vy - BH1_vy
        gasCM2BH1_vz = gasCM_vz - BH1_vz

        gasCM2BH2_vx = gasCM_vx - BH2_vx
        gasCM2BH2_vy = gasCM_vy - BH2_vy
        gasCM2BH2_vz = gasCM_vz - BH2_vz

        gasCM2BHCM_vx = gasCM_vx - BHCM_vx
        gasCM2BHCM_vy = gasCM_vy - BHCM_vy
        gasCM2BHCM_vz = gasCM_vz - BHCM_vz

        # gas velocity arrays
        vel_gas = np.array([gas_vx, gas_vy, gas_vz])
        vel_gasCM = np.array([gasCM_vx, gasCM_vy, gasCM_vz])

        # BH velocity arrays
        vel_BH1 = np.array([BH1_vx, BH1_vy, BH1_vz])
        vel_BH2 = np.array([BH2_vx, BH2_vy, BH2_vz])
        vel_BHCM = np.array([BHCM_vx, BHCM_vy, BHCM_vz])

        # relative velocity arrays
        vel_gas2BH1 = np.array([gas2BH1_vx, gas2BH1_vy, gas2BH1_vz])
        vel_gas2BH2 = np.array([gas2BH2_vx, gas2BH2_vy, gas2BH2_vz])
        vel_gas2BHCM = np.array([gas2BHCM_vx, gas2BHCM_vy, gas2BHCM_vz])

        vel_gasCM2BH1 = np.array([gasCM2BH1_vx, gasCM2BH1_vy, gasCM2BH1_vz])
        vel_gasCM2BH2 = np.array([gasCM2BH2_vx, gasCM2BH2_vy, gasCM2BH2_vz])
        vel_gasCM2BHCM = np.array([gasCM2BHCM_vx, gasCM2BHCM_vy,
                                   gasCM2BHCM_vz])

        return vel_gas, vel_BH1, vel_BH2, vel_gasCM, vel_BHCM, vel_gas_mag,\
        gasCM_vmag, vel_BH1_mag, vel_BH2_mag, vel_BHCM_mag, vel_gas2BH1,\
        vel_gas2BH2, vel_gas2BHCM, vel_gasCM2BH1, vel_gasCM2BH2, vel_gasCM2BHCM

    def get_E(self, data_pos_gas, data_vel_gas, data_pos_BH, data_vel_BH, mgas, mBH1, mBH2, mBHtotal, bool_BH1, bool_BH2):

        self.data_pos_gas = data_pos_gas
        self.data_vel_gas = data_vel_gas

        self.data_pos_BH = data_pos_BH
        self.data_vel_BH = data_vel_BH

        self.bool_BH1 = bool_BH1
        self.bool_BH2 = bool_BH2

        self.mgas = mgas
        self.mBH1 = mBH1
        self.mBH2 = mBH2
        self.mBHtotal = mBHtotal

        pos_gas, pos_BH1, pos_BH2, pos_gasCM, pos_BHCM, pos_gas2BH1,\
        pos_gas2BH2, pos_gas2BHCM,pos_gasCM2BH1, pos_gasCM2BH2,\
        pos_gasCM2BHCM,d_gas2BH1, d_gas2BH2, d_gas2BHCM, d_gasCM2BH1,\
        d_gasCM2BH2, d_gasCM2BHCM, separation =\
        self.get_pos(self.data_pos_gas, self.data_pos_BH, self.mgas, self.mBH1,
                     self.mBH2, self.mBHtotal, self.bool_BH1, self.bool_BH2)

        # obtain necessary velocity quantities
        vel_gas, vel_BH1, vel_BH2, vel_gasCM, vel_BHCM, vel_gas_mag,\
        gasCM_vmag, vel_BH1_mag, vel_BH2_mag, vel_BHCM_mag, vel_gas2BH1,\
        vel_gas2BH2, vel_gas2BHCM, vel_gasCM2BH1, vel_gasCM2BH2,\
        vel_gasCM2BHCM =\
        self.get_vel(self.data_vel_gas, self.data_vel_BH, self.mgas, self.mBH1,
                     self.mBH2, self.mBHtotal, self.bool_BH1, self.bool_BH2)

        # Calculates energies in solar units
        T_gas2BH1 = 0.5 * self.mgas * vel_gas2BH1**2
        T_gas2BH2 = 0.5 * self.mgas * vel_gas2BH2**2
        T_gas2BHCM = 0.5 * self.mgas * vel_gas2BHCM**2

        U_gas2BH1 = -self.Go * self.mgas * self.mBH1 / d_gas2BH1
        U_gas2BH2 = -self.Go * self.mgas * self.mBH2 / d_gas2BH2
        U_gas2BHCM = -self.Go * self.mgas * self.mBHtotal / d_gas2BHCM

        E_gas2BH1 = T_gas2BH1 + U_gas2BH1
        E_gas2BH2 = T_gas2BH2 + U_gas2BH2
        E_gas2BHCM = T_gas2BHCM + U_gas2BHCM

        E_gas2BH1_tot = np.sum(E_gas2BH1)
        E_gas2BH2_tot = np.sum(E_gas2BH2)
        E_gas2BHCM_tot = np.sum(E_gas2BHCM)

        bool_gas2BH1_bound = E_gas2BH1 < 0
        bool_gas2BH2_bound = E_gas2BH2 < 0
        bool_gas2BHCM_bound = E_gas2BHCM < 0

        # I get the x component bool to keep it a 1xN array
        bool_gas2BH1_bound = bool_gas2BH1_bound[0]
        bool_gas2BH2_bound = bool_gas2BH2_bound[0]
        bool_gas2BHCM_bound = bool_gas2BHCM_bound[0]

        return T_gas2BH1, T_gas2BH2, T_gas2BHCM, U_gas2BH1, U_gas2BH2,\
               U_gas2BHCM, E_gas2BH1, E_gas2BH2, E_gas2BHCM, E_gas2BH1_tot,\
               E_gas2BH2_tot, E_gas2BHCM_tot, bool_gas2BH1_bound,\
               bool_gas2BH2_bound, bool_gas2BHCM_bound

    def get_mdot(self, data_pos_gas, data_vel_gas, data_pos_BH, data_vel_BH, mgas, mBH1, mBH2, mBHtotal, bool_BH1, bool_BH2, mdot_radius):

        """Gets the mdot arrays of each BH"""

        # initializing all parameters
        self.data_pos_gas = data_pos_gas
        self.data_vel_gas = data_vel_gas

        self.bool_BH1 = bool_BH1
        self.bool_BH2 = bool_BH2

        self.data_pos_BH = data_pos_BH
        self.data_vel_BH = data_vel_BH

        self.mgas = mgas
        self.mBH1 = mBH1
        self.mBH2 = mBH2
        self.mBHtotal = mBHtotal

        self.mdot_radius = mdot_radius

        # obtain necessary position quantities
        pos_gas, pos_BH1, pos_BH2, pos_gasCM, pos_BHCM, pos_gas2BH1,\
        pos_gas2BH2, pos_gas2BHCM,pos_gasCM2BH1, pos_gasCM2BH2,\
        pos_gasCM2BHCM,d_gas2BH1, d_gas2BH2, d_gas2BHCM, d_gasCM2BH1,\
        d_gasCM2BH2, d_gasCM2BHCM, separation =\
        self.get_pos(self.data_pos_gas, self.data_pos_BH, self.mgas, self.mBH1,
                     self.mBH2, self.mBHtotal, self.bool_BH1, self.bool_BH2)

        # obtain necessary energy quantities
        T_gas2BH1, T_gas2BH2, T_gas2BHCM, U_gas2BH1, U_gas2BH2,\
        U_gas2BHCM, E_gas2BH1, E_gas2BH2, E_gas2BHCM, E_gas2BH1_tot,\
        E_gas2BH2_tot, E_gas2BHCM_tot, bool_gas2BH1_bound,\
        bool_gas2BH2_bound, bool_gas2BHCM_bound =\
        self.get_E(self.data_pos_gas, self.data_vel_gas, self.data_pos_BH,
                   self.data_vel_BH, self.mgas, self.mBH1, self.mBH2,
                   self.mBHtotal, self.bool_BH1, self.bool_BH2)

        # get bool for particles in mdot radius
        bool_mdotBH1 = d_gas2BH1 < self.mdot_radius
        bool_mdotBH2 = d_gas2BH2 < self.mdot_radius
        bool_mdotBHCM = d_gas2BHCM < self.mdot_radius

        # get bool for particles in mdot radius and bound
        bool_mdotBH1_bound = np.logical_and(bool_gas2BH1_bound,
                                            bool_mdotBH1)
        bool_mdotBH2_bound = np.logical_and(bool_gas2BH2_bound,
                                            bool_mdotBH2)
        bool_mdotBHCM_bound = np.logical_and(bool_gas2BHCM_bound,
                                             bool_mdotBHCM)

        # check which particles in the shell are bound and taken into account
        # for the new core
        mdot_BH1 = sum(self.mgas[bool_mdotBH1_bound])
        mdot_BH2 = sum(self.mgas[bool_mdotBH2_bound])
        mdot_BHCM = sum(self.mgas[bool_mdotBHCM_bound])

        # number of gas particles in mdot radius of respective BHs
        N_mdotBH1 = len(self.mgas[bool_mdotBH1])
        N_mdotBH2 = len(self.mgas[bool_mdotBH2])
        N_mdotBHCM = len(self.mgas[bool_mdotBHCM])

        # number of gas particles in mdot radius of and
        # bound to respective BHs
        N_mdotBH1_bound = len(self.mgas[bool_mdotBH1_bound])
        N_mdotBH2_bound = len(self.mgas[bool_mdotBH2_bound])
        N_mdotBHCM_bound = len(self.mgas[bool_mdotBHCM_bound])

        mdot_tot_BH1 = mdot_BH1 + self.mBH1
        mdot_tot_BH2 = mdot_BH2 + self.mBH2
        mdot_tot_BHCM = mdot_BHCM + self.mBHtotal

        return mdot_tot_BH1, mdot_tot_BH2, mdot_tot_BHCM,\
               N_mdotBH1, N_mdotBH2, N_mdotBHCM, N_mdotBH1_bound,\
               N_mdotBH2_bound, N_mdotBHCM_bound

    def get_L(self, data_pos_gas, data_vel_gas, data_pos_BH, data_vel_BH, mgas, mgas_total, mBH1, mBH2, mBHtotal, bool_BH1, bool_BH2):

        self.data_pos_gas = data_pos_gas
        self.data_vel_gas = data_vel_gas

        self.data_pos_BH = data_pos_BH
        self.data_vel_BH = data_vel_BH

        self.bool_BH1 = bool_BH1
        self.bool_BH2 = bool_BH2

        self.mgas = mgas
        self.mgas_total = mgas_total

        self.mBH1 = mBH1
        self.mBH2 = mBH2
        self.mBHtotal = mBHtotal

        # obtain necessary position quantities
        pos_gas, pos_BH1, pos_BH2, pos_gasCM, pos_BHCM, pos_gas2BH1,\
        pos_gas2BH2, pos_gas2BHCM,pos_gasCM2BH1, pos_gasCM2BH2,\
        pos_gasCM2BHCM,d_gas2BH1, d_gas2BH2, d_gas2BHCM, d_gasCM2BH1,\
        d_gasCM2BH2, d_gasCM2BHCM, separation =\
        self.get_pos(self.data_pos_gas, self.data_pos_BH, self.mgas, self.mBH1,
                     self.mBH2, self.mBHtotal, self.bool_BH1, self.bool_BH2)

        # obtain necessary velocity quantities
        vel_gas, vel_BH1, vel_BH2, vel_gasCM, vel_BHCM, vel_gas_mag,\
        gasCM_vmag, vel_BH1_mag, vel_BH2_mag, vel_BHCM_mag, vel_gas2BH1,\
        vel_gas2BH2, vel_gas2BHCM, vel_gasCM2BH1, vel_gasCM2BH2,\
        vel_gasCM2BHCM =\
        self.get_vel(self.data_vel_gas, self.data_vel_BH, self.mgas, self.mBH1,
                     self.mBH2, self.mBHtotal, self.bool_BH1, self.bool_BH2)

        #######################################
        #       Total Agular Momentum
        #######################################

        # Calculates the angular momentum of each object in
        # the simulation
        L_gas = self.mgas * np.cross(pos_gas, vel_gas, axis=0)
        L_BH1 = self.mBH1 * np.cross(pos_BH1, vel_BH1)
        L_BH2 = self.mBH2 * np.cross(pos_BH2, vel_BH2)

        # Calcules the total angular momentum of the system for
        # each component
        Lx = np.sum(L_gas[0]) + L_BH1[0] + L_BH2[0]
        Ly = np.sum(L_gas[1]) + L_BH1[1] + L_BH2[1]
        Lz = np.sum(L_gas[2]) + L_BH1[2] + L_BH2[2]

        # Calcutes the total angular momentum magnitude of the system
        L = np.sqrt(Lx**2 + Ly**2 + Lz**2)

        #######################################
        #       Relative Agular Momentum
        #######################################

        L_gas2BH1 = self.mgas * np.cross(pos_gas2BH1, vel_gas2BH1, axis=0)
        L_gas2BH2 = self.mgas * np.cross(pos_gas2BH2, vel_gas2BH2, axis=0)
        L_gas2BHCM = self.mgas * np.cross(pos_gas2BHCM, vel_gas2BHCM, axis=0)

        L_gasCM2BH1 = self.mgas_total * np.cross(pos_gasCM2BH1,
                                                 vel_gasCM2BH1)
        L_gasCM2BH2 = self.mgas_total * np.cross(pos_gasCM2BH2,
                                                 vel_gasCM2BH2)
        L_gasCM2BHCM = self.mgas_total * np.cross(pos_gasCM2BHCM,
                                                  vel_gasCM2BHCM)

        return L_gas, L_BH1, L_BH2, L, L_gas2BH1, L_gas2BH2, L_gas2BHCM, L_gasCM2BH1, L_gasCM2BH2, L_gasCM2BHCM

    def get_angle(self, data_pos_gas, data_vel_gas, data_pos_BH, data_vel_BH, mgas, mgas_total, mBH1, mBH2, mBHtotal, bool_BH1, bool_BH2, within_sphere=False, roche_bound=False, sphere_r1=None, sphere_r2=None):

        self.data_pos_gas = data_pos_gas
        self.data_vel_gas = data_vel_gas

        self.data_pos_BH = data_pos_BH
        self.data_vel_BH = data_vel_BH

        self.bool_BH1 = bool_BH1
        self.bool_BH2 = bool_BH2

        self.mgas = mgas
        self.mgas_total = mgas_total

        self.mBH1 = mBH1
        self.mBH2 = mBH2
        self.mBHtotal = mBHtotal

        self.within_sphere = within_sphere
        self.roche_bound = roche_bound
        self.sphere_r1 = sphere_r1
        self.sphere_r2 = sphere_r2

        # obtain position quantities
        pos_gas, pos_BH1, pos_BH2, pos_gasCM, pos_BHCM, pos_gas2BH1,\
        pos_gas2BH2, pos_gas2BHCM,pos_gasCM2BH1, pos_gasCM2BH2,\
        pos_gasCM2BHCM,d_gas2BH1, d_gas2BH2, d_gas2BHCM, d_gasCM2BH1,\
        d_gasCM2BH2, d_gasCM2BHCM, separation =\
        self.get_pos(self.data_pos_gas, self.data_pos_BH, self.mgas, self.mBH1,
                     self.mBH2, self.mBHtotal, self.bool_BH1, self.bool_BH2)

        # obtain velocity quantities
        vel_gas, vel_BH1, vel_BH2, vel_gasCM, vel_BHCM, vel_gas_mag,\
        gasCM_vmag, vel_BH1_mag, vel_BH2_mag, vel_BHCM_mag, vel_gas2BH1,\
        vel_gas2BH2, vel_gas2BHCM, vel_gasCM2BH1, vel_gasCM2BH2,\
        vel_gasCM2BHCM =\
        self.get_vel(self.data_vel_gas, self.data_vel_BH, self.mgas, self.mBH1,
                     self.mBH2, self.mBHtotal, self.bool_BH1, self.bool_BH2)

        # obtain angular momentum quantities
        L_gas, L_BH1, L_BH2, L, L_gas2BH1, L_gas2BH2, L_gas2BHCM, L_gasCM2BH1,\
        L_gasCM2BH2, L_gasCM2BHCM =\
        self.get_L(self.data_pos_gas, self.data_vel_gas, self.data_pos_BH,
                   self.data_vel_BH, self.mgas, self.mgas_total, self.mBH1,
                   self.mBH2, self.mBHtotal, self.bool_BH1, self.bool_BH2)

        pos_BH12BHCM = pos_BH1 - pos_BHCM
        vel_BH12BHCM = vel_BH1 - vel_BHCM

        # obtain binary orbital angular momentum
        L_bin = np.cross(pos_BH12BHCM, vel_BH12BHCM)
        L_bin_mag = np.sqrt(L_bin[0]**2 + L_bin[1]**2 + L_bin[2]**2)

        #obtain gas and gas CM relative angular momentum
        L_gas2BH1_mag = np.sqrt(L_gas2BH1[0]**2 + L_gas2BH1[1]**2 +
                                L_gas2BH1[2]**2)
        L_gas2BH2_mag = np.sqrt(L_gas2BH2[0]**2 + L_gas2BH2[1]**2 +
                                L_gas2BH2[2]**2)

        L_gasCM2BH1_mag = np.sqrt(L_gasCM2BH1[0]**2 + L_gasCM2BH1[1]**2 +
                                  L_gasCM2BH1[2]**2)
        L_gasCM2BH2_mag = np.sqrt(L_gasCM2BH2[0]**2 + L_gasCM2BH2[1]**2 +
                                  L_gasCM2BH2[2]**2)
        L_gasCM2BHCM_mag = np.sqrt(L_gasCM2BHCM[0]**2 + L_gasCM2BHCM[1]**2 +
                                   L_gasCM2BHCM[2]**2)

        L_bin_unit = L_bin/L_bin_mag

        L_gas2BH1_unit = L_gas2BH1/L_gas2BH1_mag
        L_gas2BH2_unit = L_gas2BH2/L_gas2BH2_mag

        L_gasCM2BH1_unit = L_gasCM2BH1/L_gasCM2BH1_mag
        L_gasCM2BH2_unit = L_gasCM2BH2/L_gasCM2BH2_mag
        L_gasCM2BHCM_unit = L_gasCM2BHCM/L_gasCM2BHCM_mag

        # get dot product between L_bin and relative angular momenta
        dot_gas2BH1 = L_gas2BH1_unit[0] * L_bin_unit[0] +\
                      L_gas2BH1_unit[1] * L_bin_unit[1] +\
                      L_gas2BH1_unit[2] * L_bin_unit[2]
        dot_gas2BH2 = L_gas2BH2_unit[0] * L_bin_unit[0] +\
                      L_gas2BH2_unit[1] * L_bin_unit[1] +\
                      L_gas2BH2_unit[2] * L_bin_unit[2]

        dot_gasCM2BH1 = L_gasCM2BH1_unit[0] * L_bin_unit[0] +\
                        L_gasCM2BH1_unit[1] * L_bin_unit[1] +\
                        L_gasCM2BH1_unit[2] * L_bin_unit[2]
        dot_gasCM2BH2 = L_gasCM2BH2_unit[0] * L_bin_unit[0] +\
                        L_gasCM2BH2_unit[1] * L_bin_unit[1] +\
                        L_gasCM2BH2_unit[2] * L_bin_unit[2]
        dot_gasCM2BHCM = L_gasCM2BHCM_unit[0] * L_bin_unit[0] +\
                         L_gasCM2BHCM_unit[1] * L_bin_unit[1] +\
                         L_gasCM2BHCM_unit[2] * L_bin_unit[2]

        # get angle between L_bin and relative angular momenta
        angle_gas2BH1 = np.arccos(dot_gas2BH1)
        angle_gas2BH2 = np.arccos(dot_gas2BH2)

        angle_gasCM2BH1 = np.arccos(dot_gasCM2BH1)
        angle_gasCM2BH2 = np.arccos(dot_gasCM2BH2)
        angle_gasCM2BHCM = np.arccos(dot_gasCM2BHCM)

        if within_sphere is True:

            # obtain necessary energy quantities
            T_gas2BH1, T_gas2BH2, T_gas2BHCM, U_gas2BH1, U_gas2BH2,\
            U_gas2BHCM, E_gas2BH1, E_gas2BH2, E_gas2BHCM, E_gas2BH1_tot,\
            E_gas2BH2_tot, E_gas2BHCM_tot, bool_gas2BH1_bound,\
            bool_gas2BH2_bound, bool_gas2BHCM_bound =\
            self.get_E(self.data_pos_gas, self.data_vel_gas, self.data_pos_BH,
                       self.data_vel_BH, self.mgas, self.mBH1, self.mBH2,
                       self.mBHtotal, self.bool_BH1, self.bool_BH2)

            if roche_bound is True:

                # sets larger and smaller BHs
                big_BH = max(self.mBH1,self.mBH2)
                small_BH = min(self.mBH1,self.mBH2)

                # calculates the mass ratio
                q_b_1 = small_BH/big_BH

                # sets the sphere radii to be the respective roche lobes
                sphere_r1, sphere_r2 = self.roche_lobe(q_b_1,separation)

            # boolean for particles bound within sphere
            bool_gas2BH1_sphere_bound = np.logical_and(bool_gas2BH1_bound,
                                                       d_gas2BH1 < sphere_r1)
            bool_gas2BH2_sphere_bound = np.logical_and(bool_gas2BH2_bound,
                                                       d_gas2BH2 < sphere_r2)
            bool_gas2BHCM_sphere_bound = np.logical_and(bool_gas2BHCM_bound,
                                                       d_gas2BHCM < sphere_r1)

            # individual particle angular momentum
            Lx_gas2BH1_sphere_bound_particle = L_gas2BH1[0][bool_gas2BH1_sphere_bound]
            Ly_gas2BH1_sphere_bound_particle = L_gas2BH1[1][bool_gas2BH1_sphere_bound]
            Lz_gas2BH1_sphere_bound_particle = L_gas2BH1[2][bool_gas2BH1_sphere_bound]

            Lx_gas2BH2_sphere_bound_particle = L_gas2BH2[0][bool_gas2BH2_sphere_bound]
            Ly_gas2BH2_sphere_bound_particle = L_gas2BH2[1][bool_gas2BH2_sphere_bound]
            Lz_gas2BH2_sphere_bound_particle = L_gas2BH2[2][bool_gas2BH2_sphere_bound]

            Lx_gas2BHCM_sphere_bound_particle = L_gas2BHCM[0][bool_gas2BHCM_sphere_bound]
            Ly_gas2BHCM_sphere_bound_particle = L_gas2BHCM[1][bool_gas2BHCM_sphere_bound]
            Lz_gas2BHCM_sphere_bound_particle = L_gas2BHCM[2][bool_gas2BHCM_sphere_bound]

            # summation of particle angular momentum
            Lx_gas2BH1_sphere_bound = np.sum(L_gas2BH1[0][bool_gas2BH1_sphere_bound])
            Ly_gas2BH1_sphere_bound = np.sum(L_gas2BH1[1][bool_gas2BH1_sphere_bound])
            Lz_gas2BH1_sphere_bound = np.sum(L_gas2BH1[2][bool_gas2BH1_sphere_bound])

            Lx_gas2BH2_sphere_bound = np.sum(L_gas2BH2[0][bool_gas2BH2_sphere_bound])
            Ly_gas2BH2_sphere_bound = np.sum(L_gas2BH2[1][bool_gas2BH2_sphere_bound])
            Lz_gas2BH2_sphere_bound = np.sum(L_gas2BH2[2][bool_gas2BH2_sphere_bound])

            Lx_gas2BHCM_sphere_bound = np.sum(L_gas2BHCM[0][bool_gas2BHCM_sphere_bound])
            Ly_gas2BHCM_sphere_bound = np.sum(L_gas2BHCM[1][bool_gas2BHCM_sphere_bound])
            Lz_gas2BHCM_sphere_bound = np.sum(L_gas2BHCM[2][bool_gas2BHCM_sphere_bound])

            print 'number of particles for BH1', len(L_gas2BH1[0][bool_gas2BH1_sphere_bound])

            print 'number of particles for BH2', len(L_gas2BH2[0][bool_gas2BH2_sphere_bound])

            print 'number of particles for BHCM', len(L_gas2BHCM[0][bool_gas2BHCM_sphere_bound])

            # Puts position and CM values into N x 3 arrays
            # individual
            L_gas2BH1_sphere_bound_particle = \
            np.array([Lx_gas2BH1_sphere_bound_particle,
                      Ly_gas2BH1_sphere_bound_particle,
                      Lz_gas2BH1_sphere_bound_particle])

            L_gas2BH2_sphere_bound_particle = \
            np.array([Lx_gas2BH2_sphere_bound_particle,
                      Ly_gas2BH2_sphere_bound_particle,
                      Lz_gas2BH2_sphere_bound_particle])

            L_gas2BHCM_sphere_bound_particle = \
            np.array([Lx_gas2BHCM_sphere_bound_particle,
                      Ly_gas2BHCM_sphere_bound_particle,
                      Lz_gas2BHCM_sphere_bound_particle])

            # summation
            L_gas2BH1_sphere_bound = np.array([Lx_gas2BH1_sphere_bound,
                                              Ly_gas2BH1_sphere_bound,
                                              Lz_gas2BH1_sphere_bound])

            L_gas2BH2_sphere_bound = np.array([Lx_gas2BH2_sphere_bound,
                                              Ly_gas2BH2_sphere_bound,
                                              Lz_gas2BH2_sphere_bound])

            L_gas2BHCM_sphere_bound = np.array([Lx_gas2BHCM_sphere_bound,
                                              Ly_gas2BHCM_sphere_bound,
                                              Lz_gas2BHCM_sphere_bound])

            # obtain gas and gas CM relative angular momentum of
            # bound material inside roche lobe
            # individual
            L_gas2BH1_sphere_bound_mag_particle = \
            np.sqrt(Lx_gas2BH1_sphere_bound_particle**2 +
                    Ly_gas2BH1_sphere_bound_particle**2 +
                    Lz_gas2BH1_sphere_bound_particle**2)

            L_gas2BH2_sphere_bound_mag_particle = \
            np.sqrt(Lx_gas2BH2_sphere_bound_particle**2 +
                    Ly_gas2BH2_sphere_bound_particle**2 +
                    Lz_gas2BH2_sphere_bound_particle**2)

            L_gas2BHCM_sphere_bound_mag_particle = \
            np.sqrt(Lx_gas2BHCM_sphere_bound_particle**2 +
                    Ly_gas2BHCM_sphere_bound_particle**2 +
                    Lz_gas2BHCM_sphere_bound_particle**2)

            # summation
            L_gas2BH1_sphere_bound_mag = np.sqrt(Lx_gas2BH1_sphere_bound**2 +
                                                 Ly_gas2BH1_sphere_bound**2 +
                                                 Lz_gas2BH1_sphere_bound**2)
            L_gas2BH2_sphere_bound_mag = np.sqrt(Lx_gas2BH2_sphere_bound**2 +
                                                 Ly_gas2BH2_sphere_bound**2 +
                                                 Lz_gas2BH2_sphere_bound**2)
            L_gas2BHCM_sphere_bound_mag = np.sqrt(Lx_gas2BHCM_sphere_bound**2 +
                                                  Ly_gas2BHCM_sphere_bound**2 +
                                                  Lz_gas2BHCM_sphere_bound**2)

            # get unit vectors
            # individual
            L_gas2BH1_sphere_bound_unit_particle = \
            (L_gas2BH1_sphere_bound_particle /
             L_gas2BH1_sphere_bound_mag_particle)

            L_gas2BH2_sphere_bound_unit_particle = \
            (L_gas2BH2_sphere_bound_particle /
             L_gas2BH2_sphere_bound_mag_particle)

            L_gas2BHCM_sphere_bound_unit_particle = \
            (L_gas2BHCM_sphere_bound_particle /
             L_gas2BHCM_sphere_bound_mag_particle)

            # summation
            L_gas2BH1_sphere_bound_unit = (L_gas2BH1_sphere_bound /
                                           L_gas2BH1_sphere_bound_mag)
            L_gas2BH2_sphere_bound_unit = (L_gas2BH2_sphere_bound /
                                           L_gas2BH2_sphere_bound_mag)
            L_gas2BHCM_sphere_bound_unit = (L_gas2BHCM_sphere_bound /
                                            L_gas2BHCM_sphere_bound_mag)

            # get dot product between L_bin and relative angular momenta
            #individual
            dot_gas2BH1_sphere_bound_particle = \
            (L_gas2BH1_sphere_bound_unit_particle[0] * L_bin_unit[0] +
             L_gas2BH1_sphere_bound_unit_particle[1] * L_bin_unit[1] +
             L_gas2BH1_sphere_bound_unit_particle[2] * L_bin_unit[2])

            dot_gas2BH2_sphere_bound_particle = \
            (L_gas2BH2_sphere_bound_unit_particle[0] * L_bin_unit[0] +
             L_gas2BH2_sphere_bound_unit_particle[1] * L_bin_unit[1] +
             L_gas2BH2_sphere_bound_unit_particle[2] * L_bin_unit[2])

            dot_gas2BHCM_sphere_bound_particle = \
            (L_gas2BHCM_sphere_bound_unit_particle[0] * L_bin_unit[0] +
             L_gas2BHCM_sphere_bound_unit_particle[1] * L_bin_unit[1] +
             L_gas2BHCM_sphere_bound_unit_particle[2] * L_bin_unit[2])

            # summation
            dot_gas2BH1_sphere_bound = (L_gas2BH1_sphere_bound_unit[0] *
                                        L_bin_unit[0] +
                                        L_gas2BH1_sphere_bound_unit[1] *
                                        L_bin_unit[1] +
                                        L_gas2BH1_sphere_bound_unit[2] *
                                        L_bin_unit[2])
            dot_gas2BH2_sphere_bound = (L_gas2BH2_sphere_bound_unit[0] *
                                        L_bin_unit[0] +
                                        L_gas2BH2_sphere_bound_unit[1] *
                                        L_bin_unit[1] +
                                        L_gas2BH2_sphere_bound_unit[2] *
                                        L_bin_unit[2])
            dot_gas2BHCM_sphere_bound = (L_gas2BHCM_sphere_bound_unit[0] *
                                         L_bin_unit[0] +
                                         L_gas2BHCM_sphere_bound_unit[1] *
                                         L_bin_unit[1] +
                                         L_gas2BHCM_sphere_bound_unit[2] *
                                         L_bin_unit[2])

            # get angle between L_bin and relative angular momenta
            # individual
            self.angle_gas2BH1_sphere_bound_particle = np.arccos(dot_gas2BH1_sphere_bound_particle)
            self.angle_gas2BH2_sphere_bound_particle = np.arccos(dot_gas2BH2_sphere_bound_particle)
            self.angle_gas2BHCM_sphere_bound_particle = np.arccos(dot_gas2BHCM_sphere_bound_particle)

            #summation
            self.angle_gas2BH1_sphere_bound = np.arccos(dot_gas2BH1_sphere_bound)
            self.angle_gas2BH2_sphere_bound = np.arccos(dot_gas2BH2_sphere_bound)
            self.angle_gas2BHCM_sphere_bound = np.arccos(dot_gas2BHCM_sphere_bound)

        return angle_gas2BH1, angle_gas2BH2, angle_gasCM2BH1, angle_gasCM2BH2, angle_gasCM2BHCM

    def get_mbound(self, data_pos_gas, data_vel_gas, data_pos_BH, data_vel_BH, mgas, mgas_total, mBH1, mBH2, mBHtotal, bool_BH1, bool_BH2, printinfo=False, shell_grid_res=1000, shell_radius_factor=1., Roche_Limit=False):
        """
        Obtain amount of mass bound each black hole in a binary using Davies' method.
        Receives:

        (Arguments)
        printinfo (True/False): Prints out how much mass is bound in each shell if set to True
        shell_grid_res: Sets the resolution of how many shells you want between the center of the BH and the distance you are looking out to
        Roche_Limit(True/False): If set to True then the shells will only go out to the Roche Lobe of the BH. If False the shells will go out to the seperation of the binary

        Returns:    r_mshell,cm_pos,cm_vel,cm_mass,Meb,Menv

        r_mshell (list) -> shell radius
        cm_pos (list) -> composite core center of mass
        cm_vel (list) -> composite core velocity
        cm_mass (list) -> composite core mass
        Meb (list) -> gas mass bound to core
        Menv (list) -> mass within shells radius
        """

        self.data_pos_gas = data_pos_gas
        self.data_vel_gas = data_vel_gas

        self.data_pos_BH = data_pos_BH
        self.data_vel_BH = data_vel_BH

        self.bool_BH1 = bool_BH1
        self.bool_BH2 = bool_BH2

        self.mgas = mgas
        self.mgas_total = mgas_total

        self.mBH1 = mBH1
        self.mBH2 = mBH2
        self.mBHtotal = mBHtotal

        self.printinfo = printinfo
        self.shell_grid_res = shell_grid_res
        self.shell_radius_factor = shell_radius_factor
        self.Roche_Limit = Roche_Limit


        r1_mshell_arr = []
        cm1_pos_arr = []
        cm1_vel_arr = []
        cm1_mass_arr = []

        r2_mshell_arr = []
        cm2_pos_arr = []
        cm2_vel_arr = []
        cm2_mass_arr = []

        r3_mshell_arr = []
        cm3_pos_arr = []
        cm3_vel_arr = []
        cm3_mass_arr = []

        mass1_bound_arr = []
        mass1_bound_frac_arr = []
        mass1_loss_frac_arr = []

        mass2_bound_arr = []
        mass2_bound_frac_arr = []
        mass2_loss_frac_arr = []

        mass3_bound_arr = []
        mass3_bound_frac_arr = []
        mass3_loss_frac_arr = []

        shell1_mass_arr = []
        shell2_mass_arr = []
        shell3_mass_arr = []

        # obtain position quantities
        pos_gas, pos_BH1, pos_BH2, pos_gasCM, pos_BHCM, pos_gas2BH1,\
        pos_gas2BH2, pos_gas2BHCM,pos_gasCM2BH1, pos_gasCM2BH2,\
        pos_gasCM2BHCM,d_gas2BH1, d_gas2BH2, d_gas2BHCM, d_gasCM2BH1,\
        d_gasCM2BH2, d_gasCM2BHCM, separation =\
        self.get_pos(self.data_pos_gas, self.data_pos_BH, self.mgas, self.mBH1,
                     self.mBH2, self.mBHtotal, self.bool_BH1, self.bool_BH2)

        # obtain velocity quantities
        vel_gas, vel_BH1, vel_BH2, vel_gasCM, vel_BHCM, vel_gas_mag,\
        gasCM_vmag, vel_BH1_mag, vel_BH2_mag, vel_BHCM_mag, vel_gas2BH1,\
        vel_gas2BH2, vel_gas2BHCM, vel_gasCM2BH1, vel_gasCM2BH2,\
        vel_gasCM2BHCM =\
        self.get_vel(self.data_vel_gas, self.data_vel_BH, self.mgas, self.mBH1,
                     self.mBH2, self.mBHtotal, self.bool_BH1, self.bool_BH2)

        # Define initial core mass, position and velocity
        # BH1
        mBH1_init = np.copy(self.mBH1)
        pos_BH1_init = np.copy(self.pos_BH1)
        vel_BH1_init = np.copy(self.vel_BH1)

        # BH2
        mBH2_init = np.copy(self.mBH2)
        pos_BH2_init = np.copy(self.pos_BH2)
        vel_BH2_init = np.copy(self.vel_BH2)

        # BHCM
        mBHtot_init = np.copy(self.mBHtot)
        pos_BHCM_init = np.copy(self.pos_BHCM)
        vel_BHCM_init = np.copy(self.vel_BHCM)

        # Reshape mass array to operate with position and velocity array
        mgas = self.mgas.reshape((self.mgas.size,1))

        mBH1_dyn = np.copy(mBH1_init)
        pos_BH1_dyn = np.copy(pos_BH1_init)
        vel_BH1_dyn = np.copy(vel_BH1_init)

        mBH2_dyn = np.copy(mBH2_init)
        pos_BH2_dyn = np.copy(pos_BH2_init)
        vel_BH2_dyn = np.copy(vel_BH2_init)

        mBHtot_dyn = np.copy(mBHtot_init)
        pos_BHCM_dyn = np.copy(pos_BHCM_init)
        vel_BHCM_dyn = np.copy(vel_BHCM_init)

        q = self.mBH1/self.mBH2
        p = 1/q

        if self.Roche_Limit is True:
            RL_BH1, RL_BH2 = self.roche_lobe(q_b=q, d=separation)

            RL_array = np.array([RL_BH1,RL_BH2])
            bool_largest_RL = RL_array == np.max(RL_array)

            if mBH2 > mBH1:
                RL_BH2 = RL_array[bool_largest_RL]
                RL_BH1 = RL_array[~bool_largest_RL]

            elif mBH1 > mBH2:
                RL_BH1 = RL_array[bool_largest_RL]
                RL_BH2 = RL_array[~bool_largest_RL]

            print 'Roche Lobes [1,2]: ', RL_BH1, RL_BH2

            # define shells to look for close particles
            # length of these must be the same
            r1_shell = np.linspace(0, RL_BH1, self.s_res)
            r2_shell = np.linspace(0, RL_BH2, self.s_res)
            r3_shell = np.linspace(0, shell_radius_factor*separation,
                                   shell_grid_res)

        elif Roche_Limit is False:
            # If we use the seperation then we do this one
            r1_shell = np.linspace(0,shell_radius_factor*separation,
                                   shell_grid_res)
            r2_shell = np.linspace(0,shell_radius_factor*separation,
                                   shell_grid_res)
            r3_shell = np.linspace(0,shell_radius_factor*separation,
                                   shell_grid_res)

        # Integrated mass
        # initial shell mass
        mshell_BH1 = 0.
        mshell_BH2 = 0
        mshell_BHCM = 0

        maccr_BH1 = 0.
        maccr_BH2 = 0.
        maccr_BHCM = 0.

        M1accr = [maccr_BH1]
        M2accr = [maccr_BH2]
        M3accr = [maccr_BHCM]

        r1_mshell = [0.]
        r2_mshell = [0.]
        r3_mshell = [0.]

        for i in np.arange(len(r1_shell)):

            # get quantities with respect to BH1
            gas2BH1 = pos_gas - BH1pos
            gas2BH1_d = np.linalg.norm(gas2BH1,axis=1)

            gas2BH1_v = vel_gas - BH1vel
            gas2BH1_vm = np.linalg.norm(gas2BH1_v,axis=1)

            # get quantities with respect to BH2
            gas2BH2 = gas['p'] - BH2pos
            gas2BH2_d = np.linalg.norm(gas2BH2,axis=1)

            gas2BH2_v = gas['v'] - BH2vel
            gas2BH2_vm = np.linalg.norm(gas2BH2_v,axis=1)

            # get quantities with respect to BH3
            gas2BH3 = gas['p'] - BH3pos
            gas2BH3_d = np.linalg.norm(gas2BH3,axis=1)

            gas2BH3_v = gas['v'] - BH3vel
            gas2BH3_vm = np.linalg.norm(gas2BH3_v,axis=1)

            if i < len(r1_shell) - 1:

                r1_mid = (r1_shell[i] + r1_shell[i+1])/2.
                r2_mid = (r2_shell[i] + r2_shell[i+1])/2.
                r3_mid = (r3_shell[i] + r3_shell[i+1])/2.

                # Look for particles in shells
                bool1_inshell = np.logical_and(gas2BH1_d < r1_shell[i+1],gas2BH1_d > r1_shell[i])
                bool2_inshell = np.logical_and(gas2BH2_d < r2_shell[i+1],gas2BH2_d > r2_shell[i])
                bool3_inshell = np.logical_and(gas2BH3_d < r3_shell[i+1],gas2BH3_d > r3_shell[i])

                #check which particles in this shell are bound and taken into account for the new core
                M1_shell = sum(gas['m'][bool1_inshell])
                M2_shell = sum(gas['m'][bool2_inshell])
                M3_shell = sum(gas['m'][bool3_inshell])

                M1_BH += M1_shell  # total mass in shells for BH1
                M2_BH += M2_shell  # total mass in shells for BH2
                M3_BH += M3_shell  # total mass in shells for BH2

                # particle's total energy (potential + kinetic + internal)
                Et1_s = (-BH1mass[0]/gas2BH1_d + 0.5*gas2BH1_vm**2 + gas['u'])[bool1_inshell]
                Et2_s = (-BH2mass[0]/gas2BH2_d + 0.5*gas2BH2_vm**2 + gas['u'])[bool2_inshell]
                Et3_s = (-BH3mass[0]/gas2BH3_d + 0.5*gas2BH3_vm**2 + gas['u'])[bool3_inshell]

                # Find material bound
                bound1_inshell = Et1_s < 0
                bound2_inshell = Et2_s < 0
                bound3_inshell = Et3_s < 0

                M1_accr += np.sum(gas['m'][bool1_inshell][bound1_inshell])  # total mass bound
                M2_accr += np.sum(gas['m'][bool2_inshell][bound2_inshell])  # total mass bound
                M3_accr += np.sum(gas['m'][bool3_inshell][bound3_inshell])  # total mass bound


                if printinfo:
                    print '-------------------------------------------------'
                    print len(gas['m'][bool1_inshell][bound1_inshell]), 'bound particles in shell', i,'at r= ',r1_shell[i]
                    print len(gas['m'][bool2_inshell][bound2_inshell]), 'bound particles in shell', i,'at r= ',r2_shell[i]
                    print len(gas['m'][bool3_inshell][bound3_inshell]), 'bound particles in shell', i,'at r= ',r3_shell[i]

                # get bound gas mass, momentum and position for BH1
                gas1mass = sum(gas['m'][bool1_inshell][bound1_inshell])
                gas1p = np.sum((gasm*gas['p'])[bool1_inshell][bound1_inshell],axis=0)
                gas1v = np.sum((gasm*gas['v'])[bool1_inshell][bound1_inshell],axis=0)

                # get bound gas mass, momentum and position for BH2
                gas2mass = sum(gas['m'][bool2_inshell][bound2_inshell])
                gas2p = np.sum((gasm*gas['p'])[bool2_inshell][bound2_inshell],axis=0)
                gas2v = np.sum((gasm*gas['v'])[bool2_inshell][bound2_inshell],axis=0)

                # get bound gas mass, momentum and position for BH3
                gas3mass = sum(gas['m'][bool3_inshell][bound3_inshell])
                gas3p = np.sum((gasm*gas['p'])[bool3_inshell][bound3_inshell],axis=0)
                gas3v = np.sum((gasm*gas['v'])[bool3_inshell][bound3_inshell],axis=0)


                # update BH1 CM
                BH1pos = (BH1mass*BH1pos + gas1p) / (BH1mass + gas1mass)
                BH1vel = (BH1mass*BH1vel + gas1v) / (BH1mass + gas1mass)
                BH1mass += sum(gas['m'][bool1_inshell][bound1_inshell])

                # update BH2 CM
                BH2pos = (BH2mass*BH2pos + gas2p) / (BH2mass + gas2mass)
                BH2vel = (BH2mass*BH2vel + gas2v) / (BH2mass + gas2mass)
                BH2mass += sum(gas['m'][bool2_inshell][bound2_inshell])

                # update BH3 CM
                BH3pos = (BH3mass*BH3pos + gas3p) / (BH3mass + gas3mass)
                BH3vel = (BH3mass*BH3vel + gas3v) / (BH3mass + gas3mass)
                BH3mass += sum(gas['m'][bool3_inshell][bound3_inshell])

                cm1_pos = np.vstack((cm1_pos,BH1pos))
                cm1_vel = np.concatenate((cm1_vel,BH1vel))
                cm1_mass = np.concatenate((cm1_mass,BH1mass))

                cm2_pos = np.vstack((cm2_pos,BH2pos))
                cm2_vel = np.concatenate((cm2_vel,BH2vel))
                cm2_mass = np.concatenate((cm2_mass,BH2mass))

                cm3_pos = np.vstack((cm3_pos,BH3pos))
                cm3_vel = np.concatenate((cm3_vel,BH3vel))
                cm3_mass = np.concatenate((cm3_mass,BH3mass))

                M1accr.append(M1_accr)
                M1BH.append(M1_BH)
                r1_mshell.append(r1_mid)

                M2accr.append(M2_accr)
                M2BH.append(M2_BH)
                r2_mshell.append(r2_mid)

                M3accr.append(M3_accr)
                M3BH.append(M3_BH)
                r3_mshell.append(r3_mid)

        mstar = np.sum(gas['m'])

        mass1_bound = M1accr[-1]
        shell1_mass = M1BH[-1]

        mass2_bound = M2accr[-1]
        shell2_mass = M2BH[-1]

        mass3_bound = M3accr[-1]
        shell3_mass = M3BH[-1]

        ##Appends all the values to arrays
        cm1_pos_arr.append(np.sqrt(cm1_pos[-1][0]**2 + cm1_pos[-1][1]**2 + cm1_pos[-1][2]**2))
        cm1_vel_arr.append(np.sqrt(cm1_vel[-1][0]**2 + cm1_vel[-1][1]**2 + cm1_vel[-1][2]**2))
        cm1_mass_arr.append(cm1_mass[-1])

        cm2_pos_arr.append(np.sqrt(cm2_pos[-1][0]**2 + cm2_pos[-1][1]**2 + cm2_pos[-1][2]**2))
        cm2_vel_arr.append(np.sqrt(cm2_vel[-1][0]**2 + cm2_vel[-1][1]**2 + cm2_vel[-1][2]**2))
        cm2_mass_arr.append(cm2_mass[-1])

        cm3_pos_arr.append(np.sqrt(cm3_pos[-1][0]**2 + cm3_pos[-1][1]**2 + cm3_pos[-1][2]**2))
        cm3_vel_arr.append(np.sqrt(cm3_vel[-1][0]**2 + cm3_vel[-1][1]**2 + cm3_vel[-1][2]**2))
        cm3_mass_arr.append(cm3_mass[-1])

        mass1_bound_arr.append(mass1_bound)
        shell1_mass_arr.append(shell1_mass)

        shell2_mass_arr.append(shell2_mass)
        mass2_bound_arr.append(mass2_bound)

        shell3_mass_arr.append(shell3_mass)
        mass3_bound_arr.append(mass3_bound)

        print "Done with snapshot " + str(num) + '\n'

        numsnap = np.array(numsnap)

        cm1_pos_arr = np.array(cm1_pos_arr)
        cm1_vel_arr = np.array(cm1_vel_arr)
        cm1_mass_arr = np.array(cm1_mass_arr)

        cm2_pos_arr = np.array(cm2_pos_arr)
        cm2_vel_arr = np.array(cm2_vel_arr)
        cm2_mass_arr = np.array(cm2_mass_arr)

        cm3_pos_arr = np.array(cm3_pos_arr)
        cm3_vel_arr = np.array(cm3_vel_arr)
        cm3_mass_arr = np.array(cm3_mass_arr)

        mass1_bound_arr = np.array(mass1_bound_arr)
        shell1_mass_arr = np.array(shell1_mass_arr)

        mass2_bound_arr = np.array(mass2_bound_arr)
        shell2_mass_arr = np.array(shell2_mass_arr)

        mass3_bound_arr = np.array(mass3_bound_arr)
        shell3_mass_arr = np.array(shell3_mass_arr)

        return numsnap, cm1_pos_arr, cm1_vel_arr, cm1_mass_arr, cm2_pos_arr, cm2_vel_arr, cm2_mass_arr, cm3_pos_arr, cm3_vel_arr, cm3_mass_arr, mass1_bound_arr, mass2_bound_arr, mass3_bound_arr, shell1_mass_arr, shell2_mass_arr, shell3_mass_arr, mstar, mBH1[0], mBH2[0], mBH3[0]

    #######################################
    #       Initialization Methods
    #######################################

    def init(self):

        self.data_gas_list = []
        self.data_pos_gas_list = []
        self.data_vel_gas_list = []

        self.data_BH_list = []
        self.data_pos_BH_list = []
        self.data_vel_BH_list = []

        self.bool_BH1_list = []
        self.bool_BH2_list = []

        self.mgas_list = []
        self.mBH1_list = []
        self.mBH2_list = []

        self.mgas_total_list = []
        self.mBHtotal_list = []

        for i, num in enumerate(self.numsnap):

            data_gas, data_pos_gas, data_vel_gas, data_BH, data_pos_BH,\
            data_vel_BH, bool_BH1, bool_BH2, mgas_total, mgas, mBH1, mBH2,\
            mBHtotal = self.get_init(num)

            # gas data for other methods
            self.data_gas_list.append(data_gas)
            self.data_pos_gas_list.append(data_pos_gas)
            self.data_vel_gas_list.append(data_vel_gas)

            self.data_BH_list.append(data_BH)
            self.data_pos_BH_list.append(data_pos_BH)
            self.data_vel_BH_list.append(data_vel_BH)

            # BH IDs
            self.bool_BH1_list.append(bool_BH1)
            self.bool_BH2_list.append(bool_BH2)

            # masses of each object
            self.mgas_total_list.append(mgas_total)
            self.mgas_list.append(mgas)
            self.mBH1_list.append(mBH1)
            self.mBH2_list.append(mBH2)
            self.mBHtotal_list.append(mBHtotal)

        # create arrays
        self.data_gas_array = np.array(self.data_gas_list)
        self.data_pos_gas_array = np.array(self.data_pos_gas_list)
        self.data_vel_gas_array = np.array(self.data_vel_gas_list)

        self.data_BH_array = np.array(self.data_BH_list)
        self.data_pos_BH_array = np.array(self.data_pos_BH_list)
        self.data_vel_BH_array = np.array(self.data_vel_BH_list)

        self.bool_BH1_array = np.array(self.bool_BH1_list)
        self.bool_BH2_array = np.array(self.bool_BH2_list)

        self.mgas_total_array = np.array(self.mgas_total_list)
        self.mgas_array = np.array(self.mgas_list)
        self.mBH1_array = np.array(self.mBH1_list)
        self.mBH2_array = np.array(self.mBH2_list)
        self.mBHtotal_array = np.array(self.mBHtotal_list)

        data_list = [self.data_pos_gas_array, self.data_vel_gas_array,
                     self.data_pos_BH_array, self.data_vel_BH_array,
                     self.bool_BH1_array, self.bool_BH2_array,
                     self.mgas_total_array, self.mgas_array,
                     self.mBH1_array, self.mBH2_array, self.mBHtotal_array]

        name_list = ['data_pos_gas_array', 'data_vel_gas_array',
                     'data_pos_BH_array', 'data_vel_BH_array',
                     'bool_BH1_array', 'bool_BH2_array', 'mgas_total_array',
                     'mgas_array', 'mBH1_array', 'mBH2_array',
                     'mBHtotal_array']

        print '\nInitialized: ' + str(name_list)

        return data_list, name_list

    def init_pos(self, use_init_file=False, init_file=None):

        # Position Quantities
        self.pos_gas_list = []
        self.pos_BH1_list = []
        self.pos_BH2_list = []

        self.pos_gasCM_list = []
        self.pos_BHCM_list = []

        self.pos_gas2BH1_list = []
        self.pos_gas2BH2_list = []
        self.pos_gas2BHCM_list = []

        self.pos_gasCM2BH1_list = []
        self.pos_gasCM2BH2_list = []
        self.pos_gasCM2BHCM_list = []

        self.d_gas2BH1_list = []
        self.d_gas2BH2_list = []
        self.d_gas2BHCM_list = []

        self.d_gasCM2BH1_list = []
        self.d_gasCM2BH2_list = []
        self.d_gasCM2BHCM_list = []

        self.separation_list = []

        # unique parameters
        self.use_init_file = use_init_file
        self.init_file = init_file

        if use_init_file is True:

            self.init_data = self.read_hdf5(self.init_file)

            self.data_pos_gas_list = self.init_data[3]
            self.data_pos_BH_list = self.init_data[2]
            self.mgas_list = self.init_data[9]
            self.mgas_total_array = self.init_data[10]
            self.mBH1_list = self.init_data[6]
            self.mBH2_list = self.init_data[7]
            self.mBHtotal_list = self.init_data[8]
            self.bool_BH1_list = self.init_data[0]
            self.bool_BH2_list = self.init_data[1]

        elif self.mgas_list == []:
            self.init()

        for i, snap in enumerate(self.numsnap):

            print 'reading snapshot', snap

            pos_gas, pos_BH1, pos_BH2, pos_gasCM, pos_BHCM, pos_gas2BH1,\
            pos_gas2BH2, pos_gas2BHCM,pos_gasCM2BH1, pos_gasCM2BH2,\
            pos_gasCM2BHCM,d_gas2BH1, d_gas2BH2, d_gas2BHCM, d_gasCM2BH1,\
            d_gasCM2BH2, d_gasCM2BHCM, separation =\
            self.get_pos(self.data_pos_gas_list[i], self.data_pos_BH_list[i],
                         self.mgas_list[i], self.mBH1_list[i],
                         self.mBH2_list[i], self.mBHtotal_list[i],
                         self.bool_BH1_list[i], self.bool_BH2_list[i])

            # these lists cannot be made into arrays...so far
            self.pos_gas_list.append(pos_gas)
            self.pos_BH1_list.append(pos_BH1)
            self.pos_BH2_list.append(pos_BH2)

            self.pos_gasCM_list.append(pos_gasCM)
            self.pos_BHCM_list.append(pos_BHCM)

            self.pos_gas2BH1_list.append(pos_gas2BH1)
            self.pos_gas2BH2_list.append(pos_gas2BH2)
            self.pos_gas2BHCM_list.append(pos_gas2BHCM)

            self.pos_gasCM2BH1_list.append(pos_gasCM2BH1)
            self.pos_gasCM2BH2_list.append(pos_gasCM2BH2)
            self.pos_gasCM2BHCM_list.append(pos_gasCM2BHCM)

            self.d_gas2BH1_list.append(d_gas2BH1)
            self.d_gas2BH2_list.append(d_gas2BH2)
            self.d_gas2BHCM_list.append(d_gas2BHCM)

            self.d_gasCM2BH1_list.append(d_gasCM2BH1)
            self.d_gasCM2BH2_list.append(d_gasCM2BH2)
            self.d_gasCM2BHCM_list.append(d_gasCM2BHCM)

            self.separation_list.append(separation)

        self.d_gasCM2BH1_array = np.array(self.d_gasCM2BH1_list)
        self.d_gasCM2BH2_array = np.array(self.d_gasCM2BH2_list)
        self.d_gasCM2BHCM_array = np.array(self.d_gasCM2BHCM_list)
        self.separation_array = np.array(self.separation_list)

        # some objects in this list are just too large for my current RAM
        # data_list = [self.pos_gas_list, self.pos_BH1_list, self.pos_BH2_list,
        #              self.pos_gasCM_list, self.pos_BHCM_list,
        #              self.pos_gas2BH1_list, self.pos_gas2BH2_list,
        #              self.pos_gas2BHCM_list, self.pos_gasCM2BH1_list,
        #              self.pos_gasCM2BH2_list, self.pos_gasCM2BHCM_list,
        #              self.separation_array]

        data_list = [self.pos_BH1_list, self.pos_BH2_list,
                     self.pos_gasCM_list, self.pos_BHCM_list,
                     self.pos_gasCM2BH1_list, self.pos_gasCM2BH2_list,
                     self.pos_gasCM2BHCM_list, self.d_gasCM2BH1_array,
                     self.d_gasCM2BH2_array, self.d_gasCM2BHCM_array,
                     self.separation_array]


        name_list = ['pos_BH1_list', 'pos_BH2_list', 'pos_gasCM_list',
                     'pos_BHCM_list', 'pos_gasCM2BH1_list',
                     'pos_gasCM2BH2_list', 'pos_gasCM2BHCM_list',
                     'd_gasCM2BH1_array', 'd_gasCM2BH2_array',
                     'd_gasCM2BHCM_array', 'separation_array']

        print '\nInitialized Position: ' + str(name_list)

        return data_list, name_list

    def init_vel(self, use_init_file=False, init_file=None):

        # velocity lists
        self.vel_gas_list = []
        self.vel_gas_mag_list = []

        self.vel_BH1_list = []
        self.vel_BH2_list = []

        self.vel_gasCM_list = []
        self.vel_BHCM_list = []

        self.vel_BH1_mag_list = []
        self.vel_BH2_mag_list = []
        self.vel_BHCM_mag_list = []

        # unique parameters
        self.use_init_file = use_init_file
        self.init_file = init_file

        if use_init_file is True:

            self.init_data = self.read_hdf5(self.init_file)

            self.data_vel_gas_list = self.init_data[5]
            self.data_vel_BH_list = self.init_data[4]
            self.mgas_list = self.init_data[9]
            self.mgas_total_array = self.init_data[10]
            self.mBH1_list = self.init_data[6]
            self.mBH2_list = self.init_data[7]
            self.mBHtotal_list = self.init_data[8]
            self.bool_BH1_list = self.init_data[0]
            self.bool_BH2_list = self.init_data[1]

        elif self.mgas_list == []:
            self.init()

        for i, snap in enumerate(self.numsnap):

            print 'reading snapshot', snap

            vel_gas, vel_BH1, vel_BH2, vel_gasCM, vel_BHCM, vel_gas_mag,\
            gasCM_vmag, vel_BH1_mag, vel_BH2_mag, vel_BHCM_mag, vel_gas2BH1,\
            vel_gas2BH2, vel_gas2BHCM, vel_gasCM2BH1, vel_gasCM2BH2,\
            vel_gasCM2BHCM = self.get_vel(self.data_vel_gas_list[i],
                                          self.data_vel_BH_list[i],
                                          self.mgas_list[i],
                                          self.mBH1_list[i],
                                          self.mBH2_list[i],
                                          self.mBHtotal_list[i],
                                          self.bool_BH1_list[i],
                                          self.bool_BH2_list[i])

            """ these lists cannot be made into arrays because they are of different length in one dimension...so far"""

            self.vel_gas_list.append(vel_gas)
            self.vel_gas_mag_list.append(vel_gas_mag)

            self.vel_BH1_list.append(vel_BH1)
            self.vel_BH2_list.append(vel_BH2)

            self.vel_gasCM_list.append(vel_gasCM)
            self.vel_BHCM_list.append(vel_BHCM)

            self.vel_BH1_mag_list.append(vel_BH1_mag)
            self.vel_BH2_mag_list.append(vel_BH2_mag)
            self.vel_BHCM_mag_list.append(vel_BHCM_mag)

        data_list = [self.vel_BH1_list, self.vel_BH2_list, self.vel_gasCM_list,
                     self.vel_BHCM_list, self.vel_BH1_mag_list,
                     self.vel_BH2_mag_list, self.vel_BHCM_mag_list]

        name_list = ['vel_BH1_list', 'vel_BH2_list', 'vel_gasCM_list',
                     'vel_BHCM_list', 'vel_BH1_mag_list', 'vel_BH2_mag_list',
                     'vel_BHCM_mag_list']

        print '\nInitialized Velocity: ' + str(name_list)

        return data_list, name_list

    def init_E(self, use_init_file=False, init_file=None):

        """Gets the energy of the system and boolean arrays"""

        # particle kinetic energy lists
        self.T1_list = []
        self.T_gas2BH2_list = []
        self.T_gas2BHCM_list = []

        # particle potential energy lists
        self.U_gas2BH1_list = []
        self.U_gas2BH2_list = []
        self.U_gas2BHCM_list = []

        # total particle energy lists
        self.E_gas2BH1_list = []
        self.E_gas2BH2_list = []
        self.E_gas2BHCM_list = []

        # total system energy lists
        self.E_gas2BH1_tot_list = []
        self.E_gas2BH2_tot_list = []
        self.E_gas2BHCM_tot_list = []

        # the boolean arrays for bound particles
        self.bool_gas2BH1_bound_list = []
        self.bool_gas2BH2_bound_list = []
        self.bool_gas2BHCM_bound_list = []

        # unique parameters
        self.use_init_file = use_init_file
        self.init_file = init_file

        if use_init_file is True:

            self.init_data = self.read_hdf5(self.init_file)

            self.data_pos_gas_list = self.init_data[3]
            self.data_pos_BH_list = self.init_data[2]
            self.data_vel_gas_list = self.init_data[5]
            self.data_vel_BH_list = self.init_data[4]
            self.mgas_list = self.init_data[9]
            self.mgas_total_array = self.init_data[10]
            self.mBH1_list = self.init_data[6]
            self.mBH2_list = self.init_data[7]
            self.mBHtotal_list = self.init_data[8]
            self.bool_BH1_list = self.init_data[0]
            self.bool_BH2_list = self.init_data[1]

        elif self.mgas_list == []:
            self.init()

        for i, snap in enumerate(self.numsnap):

            print 'reading snapshot', snap

            T_gas2BH1, T_gas2BH2, T_gas2BHCM, U_gas2BH1, U_gas2BH2,\
            U_gas2BHCM, E_gas2BH1, E_gas2BH2, E_gas2BHCM, E_gas2BH1_tot,\
            E_gas2BH2_tot, E_gas2BHCM_tot, bool_gas2BH1_bound,\
            bool_gas2BH2_bound, bool_gas2BHCM_bound = \
            self.get_E(self.data_pos_gas_list[i], self.data_vel_gas_list[i],
                       self.data_pos_BH_list[i], self.data_vel_BH_list[i],
                       self.mgas_list[i], self.mBH1_list[i], self.mBH2_list[i],
                       self.mBHtotal_list[i], self.bool_BH1_list[i],
                       self.bool_BH2_list[i])

            # particle kinetic energy lists
            self.T1_list.append(T_gas2BH1)
            self.T_gas2BH2_list.append(T_gas2BH2)
            self.T_gas2BHCM_list.append(T_gas2BHCM)

            # particle potential energy lists
            self.U_gas2BH1_list.append(U_gas2BH1)
            self.U_gas2BH2_list.append(U_gas2BH2)
            self.U_gas2BHCM_list.append(U_gas2BHCM)

            # total particle energy lists
            self.E_gas2BH1_list.append(E_gas2BH1)
            self.E_gas2BH2_list.append(E_gas2BH2)
            self.E_gas2BHCM_list.append(E_gas2BHCM)

            # total system energy lists
            self.E_gas2BH1_tot_list.append(E_gas2BH1_tot)
            self.E_gas2BH2_tot_list.append(E_gas2BH2_tot)
            self.E_gas2BHCM_tot_list.append(E_gas2BHCM_tot)

            # the boolean arrays for bound particles
            self.bool_gas2BH1_bound_list.append(bool_gas2BH1_bound)
            self.bool_gas2BH2_bound_list.append(bool_gas2BH2_bound)
            self.bool_gas2BHCM_bound_list.append(bool_gas2BHCM_bound)

        # creating an array
        self.E_gas2BH1_tot_array = np.array(self.E_gas2BH1_tot_list)
        self.E_gas2BH2_tot_array = np.array(self.E_gas2BH2_tot_list)
        self.E_gas2BHCM_tot_array = np.array(self.E_gas2BHCM_tot_list)

        data_list = [self.T1_list, self.T_gas2BH2_list, self.T_gas2BHCM_list,
                     self.U_gas2BH1_list, self.U_gas2BH2_list,
                     self.U_gas2BHCM_list, self.E_gas2BH1_list,
                     self.E_gas2BH2_list, self.E_gas2BHCM_list,
                     self.E_gas2BH1_tot_array, self.E_gas2BH2_tot_array,
                     self.E_gas2BHCM_tot_array, self.bool_gas2BH1_bound_list,
                     self.bool_gas2BH2_bound_list,
                     self.bool_gas2BHCM_bound_list]

        name_list = ['T1_list', 'T_gas2BH2_list', 'T_gas2BHCM_list',
                     'U_gas2BH1_list', 'U_gas2BH2_list', 'U_gas2BHCM_list',
                     'E_gas2BH1_list', 'E_gas2BH2_list', 'E_gas2BHCM_list',
                     'E_gas2BH1_tot_array', 'E_gas2BH2_tot_array',
                     'E_gas2BHCM_tot_array', 'bool_gas2BH1_bound_list',
                     'bool_gas2BH2_bound_list', 'bool_gas2BHCM_bound_list']

        print '\nInitialized Energy: ' + str(name_list)

        return data_list, name_list

    def init_mdot(self, use_init_file=False, init_file=None, mdot_radius=1, tdyn_per_snap=1):

        """

        Gets total and relative angular momenta of the entire system

        mdot_radius = sets the accretion radius [Ro]

        """

        # initialize parameters
        self.mdot_radius = mdot_radius
        self.tdyn_per_snap = tdyn_per_snap

        # mdot lists
        self.N_mdotBH1_list = []
        self.N_mdotBH2_list = []
        self.N_mdotBHCM_list = []

        self.N_mdotBH1_bound_list = []
        self.N_mdotBH2_bound_list = []
        self.N_mdotBHCM_bound_list = []

        self.mdot_tot_BH1_list = []
        self.mdot_tot_BH2_list = []
        self.mdot_tot_BHCM_list = []

        # unique parameters
        self.use_init_file = use_init_file
        self.init_file = init_file

        if use_init_file is True:

            self.init_data = self.read_hdf5(self.init_file)

            self.data_pos_gas_list = self.init_data[3]
            self.data_pos_BH_list = self.init_data[2]
            self.data_vel_gas_list = self.init_data[5]
            self.data_vel_BH_list = self.init_data[4]
            self.mgas_list = self.init_data[9]
            self.mgas_total_array = self.init_data[10]
            self.mBH1_list = self.init_data[6]
            self.mBH2_list = self.init_data[7]
            self.mBHtotal_list = self.init_data[8]
            self.bool_BH1_list = self.init_data[0]
            self.bool_BH2_list = self.init_data[1]

        elif self.mgas_list == []:
            self.init()

        for i, snap in enumerate(self.numsnap):

            print 'reading snapshot', snap

            # obtain mdot quantities
            mdot_tot_BH1, mdot_tot_BH2, mdot_tot_BHCM,\
            N_mdotBH1, N_mdotBH2, N_mdotBHCM, N_mdotBH1_bound,\
            N_mdotBH2_bound, N_mdotBHCM_bound =\
            self.get_mdot(self.data_pos_gas_list[i], self.data_vel_gas_list[i],
                          self.data_pos_BH_list[i], self.data_vel_BH_list[i],
                          self.mgas_list[i], self.mBH1_list[i],
                          self.mBH2_list[i], self.mBHtotal_list[i],
                          self.bool_BH1_list[i], self.bool_BH2_list[i],
                          self.mdot_radius)

            self.N_mdotBH1_list.append(N_mdotBH1)
            self.N_mdotBH2_list.append(N_mdotBH2)
            self.N_mdotBHCM_list.append(N_mdotBHCM)

            self.N_mdotBH1_bound_list.append(N_mdotBH1_bound)
            self.N_mdotBH2_bound_list.append(N_mdotBH2_bound)
            self.N_mdotBHCM_bound_list.append(N_mdotBHCM_bound)

            self.mdot_tot_BH1_list.append(mdot_tot_BH1)
            self.mdot_tot_BH2_list.append(mdot_tot_BH2)
            self.mdot_tot_BHCM_list.append(mdot_tot_BHCM)

        dm_BH1 = np.diff(self.mdot_tot_BH1_list)
        dm_BH2 = np.diff(self.mdot_tot_BH2_list)
        dm_BHCM = np.diff(self.mdot_tot_BHCM_list)

        # calculates the time between indexes
        dt = (self.numsnap[1] - self.numsnap[0]) *\
              self.tdyn_per_snap * self.sec_per_tdyn

        self.dmdt_BH1_array = dm_BH1 / dt
        self.dmdt_BH2_array = dm_BH2 / dt
        self.dmdt_BHCM_array = dm_BHCM / dt

        # add a zero or else it will be one element less than other lists
        self.dmdt_BH1_array = np.concatenate((self.dmdt_BH1_array, [0]))
        self.dmdt_BH2_array = np.concatenate((self.dmdt_BH2_array, [0]))
        self.dmdt_BHCM_array = np.concatenate((self.dmdt_BHCM_array, [0]))

        # creating an array
        self.N_mdotBH1_array = np.array(self.N_mdotBH1_list)
        self.N_mdotBH2_array = np.array(self.N_mdotBH2_list)
        self.N_mdotBHCM_array = np.array(self.N_mdotBHCM_list)

        self.N_mdotBH1_bound_array = np.array(self.N_mdotBH1_bound_list)
        self.N_mdotBH2_bound_array = np.array(self.N_mdotBH2_bound_list)
        self.N_mdotBHCM_bound_array = np.array(self.N_mdotBHCM_bound_list)

        self.mdot_tot_BH1_array = np.array(self.mdot_tot_BH1_list)
        self.mdot_tot_BH2_array = np.array(self.mdot_tot_BH2_list)
        self.mdot_tot_BHCM_array = np.array(self.mdot_tot_BHCM_list)

        data_list = [self.N_mdotBH1_array, self.N_mdotBH2_array,
                     self.N_mdotBHCM_array, self.N_mdotBH1_bound_array,
                     self.N_mdotBH2_bound_array, self.N_mdotBHCM_bound_array,
                     self.mdot_tot_BH1_array, self.mdot_tot_BH2_array,
                     self.mdot_tot_BHCM_array, self.dmdt_BH1_array,
                     self.dmdt_BH2_array, self.dmdt_BHCM_array]

        name_list = ['N_mdotBH1_array', 'N_mdotBH2_array', 'N_mdotBHCM_array',
                     'N_mdotBH1_bound_array', 'N_mdotBH2_bound_array',
                     'N_mdotBHCM_bound_array', 'mdot_tot_BH1_array',
                     'mdot_tot_BH2_array', 'mdot_tot_BHCM_array',
                     'dmdt_BH1_array', 'dmdt_BH2_array', 'dmdt_BHCM_array']

        print '\nInitialized mdot: ' + str(name_list)

        return data_list, name_list

    def init_L(self, use_init_file=False, init_file=None):

        """Gets total and relative angular momenta of the entire system"""

        # angular momentum lists
        self.L_gas_list = []
        self.L_BH1_list = []
        self.L_BH2_list = []

        self.L_total_list = []

        self.L_gas2BH1_list = []
        self.L_gas2BH2_list = []
        self.L_gas2BHCM_list = []

        self.L_gasCM2BH1_list = []
        self.L_gasCM2BH2_list = []
        self.L_gasCM2BHCM_list = []

        # unique parameters
        self.use_init_file = use_init_file
        self.init_file = init_file

        if use_init_file is True:

            self.init_data = self.read_hdf5(self.init_file)

            self.data_pos_gas_list = self.init_data[3]
            self.data_pos_BH_list = self.init_data[2]
            self.data_vel_gas_list = self.init_data[5]
            self.data_vel_BH_list = self.init_data[4]
            self.mgas_list = self.init_data[9]
            self.mgas_total_array = self.init_data[10]
            self.mBH1_list = self.init_data[6]
            self.mBH2_list = self.init_data[7]
            self.mBHtotal_list = self.init_data[8]
            self.bool_BH1_list = self.init_data[0]
            self.bool_BH2_list = self.init_data[1]

        elif self.mgas_list == []:
            self.init()

        for i, snap in enumerate(self.numsnap):

            print 'reading snapshot', snap

            # obtain angular momentum quantities
            L_gas, L_BH1, L_BH2, L, L_gas2BH1, L_gas2BH2, L_gas2BHCM,\
            L_gasCM2BH1, L_gasCM2BH2, L_gasCM2BHCM =\
            self.get_L(self.data_pos_gas_list[i], self.data_vel_gas_list[i],
                       self.data_pos_BH_list[i], self.data_vel_BH_list[i],
                       self.mgas_list[i], self.mgas_total_array[i],
                       self.mBH1_list[i], self.mBH2_list[i],
                       self.mBHtotal_list[i], self.bool_BH1_list[i],
                       self.bool_BH2_list[i])

            # updating angular momentum lists
            self.L_gas_list.append(L_gas)
            self.L_BH1_list.append(L_BH1)
            self.L_BH2_list.append(L_BH2)

            # updating total angular momentum lists
            self.L_total_list.append(L)

            # updating relative angular momentum lists
            self.L_gas2BH1_list.append(L_gas2BH1)
            self.L_gas2BH2_list.append(L_gas2BH2)
            self.L_gas2BHCM_list.append(L_gas2BHCM)

            self.L_gasCM2BH1_list.append(L_gasCM2BH1)
            self.L_gasCM2BH2_list.append(L_gasCM2BH2)
            self.L_gasCM2BHCM_list.append(L_gasCM2BHCM)

        # creating an array
        self.L_total_array = np.array(self.L_total_list)
        self.L_gasCM2BH1_array = np.array(self.L_gasCM2BH1_list)
        self.L_gasCM2BH2_array = np.array(self.L_gasCM2BH2_list)
        self.L_gasCM2BHCM_array = np.array(self.L_gasCM2BHCM_list)

        data_list = [self.L_total_array, self.L_gas_list, self.L_BH1_list,
                     self.L_BH2_list, self.L_gas2BH1_list, self.L_gas2BH2_list,
                     self.L_gas2BHCM_list, self.L_gasCM2BH1_array,
                     self.L_gasCM2BH2_array, self.L_gasCM2BHCM_array]

        name_list = ['L_total_array', 'L_gas_list', 'L_BH1_list', 'L_BH2_list',
                     'L_gas2BH1_list', 'L_gas2BH2_list', 'L_gas2BHCM_list',
                     'L_gasCM2BH1_array', 'L_gasCM2BH2_array',
                     'L_gasCM2BHCM_array']

        print '\nInitialized Angular Momentum: ' + str(name_list)

        return data_list, name_list

    def init_angle(self, use_init_file=False, init_file=None, within_sphere=False, roche_bound=False, sphere_r1=None, sphere_r2=None):

        """Gets the energy of the system and boolean arrays"""

        self.angle_gas2BH1_list = []
        self.angle_gas2BH2_list = []

        self.angle_gasCM2BH1_list = []
        self.angle_gasCM2BH2_list = []
        self.angle_gasCM2BHCM_list = []

        # unique parameters
        self.use_init_file = use_init_file
        self.init_file = init_file
        self.within_sphere = within_sphere
        self.roche_bound = roche_bound
        self.sphere_r1 = sphere_r1
        self.sphere_r2 = sphere_r2

        if use_init_file is True:

            self.init_data = self.read_hdf5(self.init_file)

            self.data_pos_gas_list = self.init_data[3]
            self.data_pos_BH_list = self.init_data[2]
            self.data_vel_gas_list = self.init_data[5]
            self.data_vel_BH_list = self.init_data[4]
            self.mgas_list = self.init_data[9]
            self.mgas_total_array = self.init_data[10]
            self.mBH1_list = self.init_data[6]
            self.mBH2_list = self.init_data[7]
            self.mBHtotal_list = self.init_data[8]
            self.bool_BH1_list = self.init_data[0]
            self.bool_BH2_list = self.init_data[1]

        elif self.mgas_list == []:
            self.init()

        for i, snap in enumerate(self.numsnap):

            print 'reading snapshot', snap

            angle_gas2BH1, angle_gas2BH2, angle_gasCM2BH1, angle_gasCM2BH2,\
            angle_gasCM2BHCM = \
            self.get_angle(self.data_pos_gas_list[i],
                           self.data_vel_gas_list[i], self.data_pos_BH_list[i],
                           self.data_vel_BH_list[i], self.mgas_list[i],
                           self.mgas_total_array[i], self.mBH1_list[i],
                           self.mBH2_list[i], self.mBHtotal_list[i],
                           self.bool_BH1_list[i], self.bool_BH2_list[i],
                           self.within_sphere, self.roche_bound,
                           self.sphere_r1, self.sphere_r2)

            self.angle_gas2BH1_list.append(angle_gas2BH1)
            self.angle_gas2BH2_list.append(angle_gas2BH2)

            self.angle_gasCM2BH1_list.append(angle_gasCM2BH1)
            self.angle_gasCM2BH2_list.append(angle_gasCM2BH2)
            self.angle_gasCM2BHCM_list.append(angle_gasCM2BHCM)

        # creating an array
        self.angle_gas2BH1_array = np.array(self.angle_gas2BH1_list)
        self.angle_gas2BH2_array = np.array(self.angle_gas2BH2_list)

        self.angle_gasCM2BH1_array = np.array(self.angle_gasCM2BH1_list)
        self.angle_gasCM2BH2_array = np.array(self.angle_gasCM2BH2_list)
        self.angle_gasCM2BHCM_array = np.array(self.angle_gasCM2BHCM_list)

        data_list = [self.angle_gas2BH1_array, self.angle_gas2BH2_array,
                     self.angle_gasCM2BH1_array, self.angle_gasCM2BH2_array,
                     self.angle_gasCM2BHCM_array]

        name_list = ['angle_gas2BH1_array', 'angle_gas2BH2_array',
                     'angle_gasCM2BH1_array', 'angle_gasCM2BH2_array',
                     'angle_gasCM2BHCM_array']

        print '\n============================================================='
        print 'Initialized Angles: ' + str(name_list)
        print '============================================================='

        return data_list, name_list

    #######################################
    #       Write HDF5 Methods
    #######################################

    def write_hdf5(self, write_file, list_type,use_init_file=False, init_file=None, mdot_radius=1, tdyn_per_snap=1):

        self.write_file = write_file
        self.list_type = list_type
        self.use_init_file = use_init_file
        self.init_file = init_file

        # these are only used for list_type='mdot'
        self.mdot_radius = mdot_radius
        self.tdyn_per_snap = tdyn_per_snap

        if self.list_type == 'init':
            hdf5_data_list, hdf5_name_list = self.init()

        elif self.list_type == 'pos':
            hdf5_data_list, hdf5_name_list = self.init_pos(self.use_init_file,
                                                           self.init_file)

        elif self.list_type == 'vel':
            hdf5_data_list, hdf5_name_list = self.init_vel(self.use_init_file,
                                                           self.init_file)

        elif self.list_type == 'mdot':
            hdf5_data_list, hdf5_name_list = self.init_mdot(self.use_init_file,
                                                            self.init_file,
                                                            self.mdot_radius,
                                                            self.tdyn_per_snap)

        elif self.list_type == 'L':
            hdf5_data_list, hdf5_name_list = self.init_L(self.use_init_file,
                                                         self.init_file)

        elif self.list_type == 'E':
            hdf5_data_list, hdf5_name_list = self.init_E(self.use_init_file,
                                                         self.init_file)

        elif self.list_type == 'angle':
            hdf5_data_list, hdf5_name_list =self.init_angle(self.use_init_file,
                                                            self.init_file)

        self.file = h.File(name=self.write_file, mode='w')

        for i, keyword in enumerate(hdf5_name_list):
            for j, snap in enumerate(self.numsnap):
                self.file.create_dataset(name=hdf5_name_list[i] + '/' +\
                                         str(snap), data=hdf5_data_list[i][j])
        h.File.close(self.file)

        print '\nwrite file ' + self.write_file

    #######################################
    #       Read HDF5 Methods
    #######################################

    def read_hdf5(self, read_file):

        self.dataset_list = []

        self.read_file = read_file

        self.file = h.File(name=self.read_file, mode='r')

        self.key_list = self.file.keys()

        self.dataset_list = []

        for i, key in enumerate(self.key_list):
            self.key_list = []
            print key, 'is index', i
            for j, snap in enumerate(self.numsnap):
                self.key_list.append(np.array(self.file[key].get(str(snap))))
            self.dataset_list.append(self.key_list)

        h.File.close(self.file)

        print '\nread file ' + self.read_file

        return self.dataset_list

    #######################################
    #       Misc Methods
    #######################################

    def bin_dmdt(self, dmdt, Nbin):

        self.dmdt = np.array(dmdt[:-1])
        self.Nbin = Nbin

        tlog = np.log10(self.numsnap[1:])

        edges = stats.binned_statistic(tlog, self.dmdt, bins=self.Nbin)
        dmdt_smooth = edges[0]
        numsnap_smooth = edges[1][:-1]

        M_integ = [integrate.trapz(self.dmdt,
                                   self.numsnap[1:] * self.sec_per_tdyn)]

        smooth_bool = dmdt_smooth > 0
        dmdt_smooth = dmdt_smooth[smooth_bool]
        numsnap_smooth = numsnap_smooth[smooth_bool]

        M_integ = np.array(M_integ * len(dmdt_smooth))

        dmdt_col_names = ['numsnap_smooth', 'dmdt_smooth', 'M_integ']
        dmdt_table = Table([numsnap_smooth, dmdt_smooth,
                            M_integ], names=dmdt_col_names)

        return dmdt_table

    def find_nearest(self, array, value):
        '''
        Returns index of nearest number to value in array
        '''
        self.array = np.array(array)
        self.value = value

        idx = (np.abs(self.array-self.value)).argmin()
        return idx

    def isco_stuff(self, a, Mbh):

        self.a = a
        self.Mbh = Mbh

        z1 = 1. + ((1. - a**2.)**(1./3.)) * ((1. + a)**(1./3.) +\
             (1. - a)**(1./3.))
        z2 = (3. * a**2. + z1**2.)**(0.5)

        r_isco = (3. + z2 - np.sqrt((3. - z1) * (3. + z1 + 2. * z2)))
        j_isco = (self.G * self.Mbh / self.c) *\
                 (r_isco**2. - 2. * a * r_isco**(0.5) + a**2.)/\
                 (r_isco**(3./4.)*\
                 (r_isco**(3./2.) - (3. * r_isco**(0.5)) + (2. * a))**(0.5))

        # Fraction of disk's mass accreted by BH
        e = np.sqrt(1. - (2. / (3 * r_isco)))
        # 1-e of the rest mass is "radiated" away

        return r_isco, j_isco, e

    def roche_lobe(self, q_b, d):

        self.q_b = q_b
        self.d = d
        q_b_2 = 1./self.q_b

        RL_1 = ((0.49 * self.q_b**(2./3.))/(0.6*self.q_b**(2./3.) +
                       np.log(1. + self.q_b**(1./3.))))*self.d
        RL_2 = ((0.49 * q_b_2**(2./3.))/(0.6*q_b_2**(2./3.) +
                       np.log(1. + q_b_2**(1./3.))))*self.d

        return RL_1, RL_2


def find_nearest(array, value):
    '''
    Returns index of nearest number to value in array
    '''
    array = np.array(array)

    idx = (np.abs(array-value)).argmin()
    return idx

def isco_stuff(a, Mbh, dp):
    """
        Calculate the left-sided derivative approximation at a point.

        Parameters
        ----------
        a : float
            the current dimensionless spin of the BH
        Mbh : float
            the mass of the BH
        dp : float
            the dot product between the angular momentum of the disk of mass which is being accreted and the current spin of the BH

        Returns
        ----------
        r_isco: float
            the radius of the inner-most stable circular orbit
        j_isco: float
            the angular momentum of the inner-most stable circular orbit
        e: float
            the fraction of disk's mass accreted by BH, rest of disk
            is "radiated" away
    """

    z1 = (1.
        + ((1. - a**2.)**(1. / 3.))
        * ((1. + a)**(1. / 3.) + (1. - a)**(1. / 3.)))

    z2 = (3. * a**2. + z1**2.)**(1. / 2.)

    if dp >= 0:
        r_isco = (3. + z2 - np.sqrt((3. - z1) * (3. + z1 + 2. * z2)))

    if dp < 0:
        r_isco = (3. + z2 + np.sqrt((3. - z1) * (3. + z1 + 2. * z2)))

    j_isco = ((G * Mbh / c)
             * (r_isco**2.
                - 2. * a * r_isco**(0.5)
                + a**2.)
             / (r_isco**(3. / 4.) * (r_isco**(3. / 2.)
                - 3. * r_isco**(0.5)
                + 2. * a)**(0.5)))

    # Fraction of disk's mass accreted by BH
    e = np.sqrt(1. - 2. / (3 * r_isco))
    # 1-e of the rest mass is "radiated" away

    return r_isco, j_isco, e

def r_tau(Mstar, Rstar, Mbh):
    r = Rstar * (Mbh / Mstar)**(1. / 3.)
    return r

def c_l_scales(Rstar, Mstar, mBH1, mBH2, d, R_tau, f_half=0.28, f_90=0.13, print_out=False, tolerance=1.e-3):
    # Solar Constants
    Ro = 6.955e10                         # Solar Radius [cm]
    Mo = 1.989e33                        # Solar Mass [g]

    q = mBH1 / mBH2
    mBH = mBH1
    ave_sep = d/2.
    min_sep = d/4.

    RLR = (((0.49 * q**(2. / 3.)) /\
           (0.6 * q **(2. / 3.) + np.log(1. + q**(1. / 3.)))) * min_sep)

    a_mb_1 = a_n(Rstar, Mstar, mBH, Rstar, n=1)
    a_mb = a_n(Rstar, Mstar, mBH, Rstar, convergence=True,
               tolerance=tolerance, print_out=print_out)

    a_half_1 = a_n(Rstar, Mstar, mBH, f_half * Rstar, n=1)
    a_half = a_n(Rstar, Mstar, mBH, f_half * Rstar, convergence=True,
                 tolerance=tolerance, print_out=print_out)

    a_90_1 = a_n(Rstar, Mstar, mBH, f_90 * Rstar, n=1)
    a_90 = a_n(Rstar, Mstar, mBH, f_90 * Rstar, convergence=True,
               tolerance=tolerance, print_out=print_out)

    psi_mb = 0.5 * (1 + R_tau / d - Rstar / d)
    psi_half = 0.5 * (1 + R_tau / d - f_half * Rstar / d)
    psi_90 = 0.5 * (1 + R_tau / d - f_90 * Rstar / d)

    print '...........................................................................................................................................'
    print '                                       Characteristic Length Scales [Ro]                                                         '
    print '...........................................................................................................................................'

    print '\n                                       ---------------------------------'
    print '                                           Using Most Bound Material'
    print '                                       ---------------------------------'

    print 'Rstar:', round(Rstar / Ro, 2)
    print 'R_tau:', round(R_tau / Ro, 2)
    print 'd:', round(d / Ro, 2)
    print 'average separation:', round(ave_sep / Ro, 2)
    print 'minimum separation:', round(min_sep / Ro, 2)
    print 'RLR:', round(RLR / Ro, 2)
    print 'psi', round(psi_mb, 2)

    print '\n1st order a_mb:', round(a_mb_1 / Ro, 2)
    print str(a_mb[1]) + 'th order a_mb:', round(a_mb[0] / Ro, 2)

    print '\n*********************************'

    print '\nR_tau/d:', round(R_tau / d, 2)

    print '\n1st order a_mb/RLR:', round(a_mb_1 / RLR, 2)

    print '1st order a_mb/d:', round(a_mb_1 / d, 2)

    print '\n' + str(a_mb[1]) + 'th order a_mb/RLR:', round(a_mb[0] / RLR, 2)

    print str(a_mb[1]) + 'th order a_mb/d:', round(a_mb[0] / d, 2)

    print '\n*********************************\n'

    if R_tau / d >= 1:
        print 'R_tau/d >= 1'

    elif R_tau / d <= 1:
        print 'R_tau/d <= 1'

    if a_mb_1 / RLR <= 1:
        print '\n1st order a_mb/RLR <= 1'

    elif a_mb_1 / RLR > 1:
        print '\n1st order a_mb/RLR > 1'

    if a_mb_1 / d < psi_mb:
        print '1st order a_mb/d < psi'
    elif a_mb_1 / d >= psi_mb:
        print '1st order a_mb/d >= psi'

    if a_mb[0] / RLR <= 1:
        print '\n' + str(a_mb[1]) + 'th order a_mb/RLR <= 1'

    elif a_mb[0] / RLR > 1:
        print '\n' + str(a_mb[1]) + 'th order a_mb/RLR > 1'

    if a_mb[0] / d < psi_mb:
        print str(a_mb[1]) + 'th order a_mb/d < psi'

    elif a_mb[0] / d >= psi_mb:
        print str(a_mb[1]) + 'th order a_mb/d >= psi'

    print '\n                                       ---------------------------------'
    print '                                            Using Half Mass Material'
    print '                                       ---------------------------------'

    print 'Rstar:', round(Rstar / Ro, 2)
    print 'R_tau:', round(R_tau / Ro, 2)
    print 'd:', round(d / Ro, 2)
    print 'average separation:', round(ave_sep / Ro, 2)
    print 'minimum separation:', round(min_sep / Ro, 2)
    print 'RLR:', round(RLR / Ro, 2)
    print 'psi', round(psi_half, 2)

    print '\n1st order a_half:', round(a_half_1 / Ro, 2)
    print str(a_half[1]) + 'th order a_half:', round(a_half[0] / Ro, 2)

    print '\n*********************************'

    print '\nR_tau/d:', round(R_tau / d, 2)

    print '\n1st order a_half/RLR:', round(a_half_1 / RLR, 2)

    print '1st order a_half/d:', round(a_half_1 / d, 2)

    print '\n' + str(a_half[1]) + 'th order a_half/RLR:', round(a_half[0] / RLR, 2)

    print str(a_half[1]) + 'th order a_half/d:', round(a_half[0] / d, 2)

    print '\n*********************************\n'

    if R_tau / d >= 1:
        print 'R_tau/d >= 1'

    elif R_tau / d <= 1:
        print 'R_tau/d <= 1'

    if a_half_1 / RLR <= 1:
        print '\n1st order a_half/RLR <= 1'

    elif a_half_1 / RLR > 1:
        print '\n1st order a_half/RLR > 1'

    if a_half_1 / d < psi_half:
        print '1st order a_half/d < psi'
    elif a_half_1 / d >= psi_half:
        print '1st order a_half/d >= psi'

    if a_half[0] / RLR <= 1:
        print '\n' + str(a_half[1]) + 'th order a_half/RLR <= 1'

    elif a_half[0] / RLR > 1:
        print '\n' + str(a_half[1]) + 'th order a_half/RLR > 1'

    if a_half[0] / d < psi_half:
        print str(a_half[1]) + 'th order a_half/d < psi'

    elif a_half[0] / d >= psi_half:
        print str(a_half[1]) + 'th order a_half/d >= psi'

    print '\n                                       ---------------------------------'
    print '                                        Using 90 Percent Mass Material'
    print '                                       ---------------------------------'

    print 'Rstar:', round(Rstar / Ro, 2)
    print 'R_tau:', round(R_tau / Ro, 2)
    print 'd:', round(d / Ro, 2)
    print 'average separation:', round(ave_sep / Ro, 2)
    print 'minimum separation:', round(min_sep / Ro, 2)
    print 'RLR:', round(RLR / Ro, 2)
    print 'psi', round(psi_90, 2)

    print '\n1st order a_90:', round(a_90_1 / Ro, 2)
    print str(a_90[1]) + 'th order a_90:', round(a_90[0] / Ro, 2)

    print '\n*********************************'

    print '\nR_tau/d:', round(R_tau / d, 2)

    print '\n1st order a_90/RLR:', round(a_90_1 / RLR, 2)

    print '1st order a_90/d:', round(a_90_1 / d, 2)

    print '\n' + str(a_90[1]) + 'th order a_90/RLR:', round(a_90[0] / RLR, 2)

    print str(a_90[1]) + 'th order a_90/d:', round(a_90[0] / d, 2)

    print '\n*********************************\n'

    if R_tau / d >= 1:
        print 'R_tau/d >= 1'

    elif R_tau / d <= 1:
        print 'R_tau/d <= 1'

    if a_90_1 / RLR <= 1:
        print '\n1st order a_90/RLR <= 1'

    elif a_90_1 / RLR > 1:
        print '\n1st order a_90/RLR > 1'

    if a_90_1 / d < psi_90:
        print '1st order a_90/d < psi'
    elif a_90_1 / d >= psi_90:
        print '1st order a_90/d >= psi'

    if a_90[0] / RLR <= 1:
        print '\n' + str(a_90[1]) + 'th order a_90/RLR <= 1'

    elif a_90[0] / RLR > 1:
        print '\n' + str(a_90[1]) + 'th order a_90/RLR > 1'

    if a_90[0] / d < psi_90:
        print str(a_90[1]) + 'th order a_90/d < psi'

    elif a_90[0] / d >= psi_90:
        print str(a_90[1]) + 'th order a_90/d >= psi'

def a_n(Rstar, Mstar, mBH, r, n=1, convergence=False, tolerance=1.e-3, print_out=False):
    '''
        Calculates the semi-major axis of material as a function of r distance
        away from the center of the star which is on a parabolic orbit being
        tidally disrupted by a BH.

        You can use any units you'd like just be consistent and the returned semi-major axis will be in those units.

        Rstar: Radius of the star being tidally disrupted.
        Mstar: Mass of the star being tidally disrupted.
        mBH: Mass of the BH causing the tidal disruption.
        r: The distance from the center of the star that you want to obtain the semi-major axis for.
        n: How many terms do you want in the taylor expansion of the semi-major axis.
    '''

    # Solar Constants
    Ro = 6.955e10                         # Solar Radius [cm]
    Mo = 1.989e33                        # Solar Mass [g]

    # initial value of the summation
    a_sum = 0

    # mass ratio
    q = Mstar/mBH

    if convergence == True:
        # initial n value
        n_i = 0

        # initial a_value
        a_i = 1e100

        diff = 100

        while diff > tolerance:
            # We start here so that the final n_i value will be the true n value of the expansion
            n_i += 1

            # This is the n_i'th term in the expansion
            a_index = (q**(1./3.) / Rstar)**(n_i + 1.) * r**n_i

            # This is the total sum of the expansion
            a_sum += a_index

            # This is the true value of a for n_i'th order expansion
            a_f = 1. / (2*a_sum)

            # We compare the expansion of n_i'th - 1 term to the n_i'th term in units of solar radii
            diff = (a_i - a_f) / Ro

            # We make the n_i'th -1 term equal to the n_i'th term for the next iteration if it occurs
            a_i = a_f

        if print_out == True:
            print 'a converged after ', n_i, ' terms'
        return [a_i, n_i]

    elif convergence == False:
        for i, index in enumerate(np.arange(1, n + 1)):
            a_index = 2. * (Mstar / mBH)**((index + 1.) / 3.) * \
                (1. / (Rstar**(index + 1.))) * r**index
            a_sum += a_index
        a = 1. / a_sum
        return a

def T_n(Rstar, Mstar, mBH, r, n):
    T_sum = 0
    for i, index in enumerate(np.arange(1, n + 1)):
        T_index = 2. * (Mstar / mBH)**((index + 1.) / 3.) * \
            (1. / (Rstar**(index + 1.))) * r**index
        T_sum += T_index
    T = 2 * np.pi * (G * (Mstar + mBH))**(-1. / 2.) * (T_sum)**(-3. / 2.)
    return T

def r_tau_over_a_r_n(q, Rstar, r, n):
    ratio_sum = 0
    for i, index in enumerate(np.arange(1, n + 1)):
        ratio_index = 2. * (q)**(index / 3.) * \
            (1. / (Rstar**(index))) * r**index
        ratio_sum += ratio_index
    return ratio_sum


