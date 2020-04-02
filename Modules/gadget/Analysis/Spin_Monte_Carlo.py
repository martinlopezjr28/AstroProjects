########################################
#                HEADER
########################################

import numpy as np
import BBH_Analysis as BBH

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

class Spin_MC(Constants):

    def __init__(self, N_MC, N_trials, Mbh1_i, Mbh2_i, spin1_i, spin2_i, Mstar, Rstar, dM):

        #######################################
        #           GET ARGUMENTS
        #######################################

        # number of MC runs
        self.N_MC = N_MC

        # number of disruptions per MC run
        self.N_trials = N_trials

        # initial BH masses
        self.Mbh1_i = Mbh1_i
        self.Mbh2_i = Mbh2_i

        # initial spin magnitudes
        self.spin1_i = spin1_i
        self.spin2_i = spin2_i

        # stellar parameters
        self.Mstar = Mstar
        self.Rstar = Rstar

        # mass accreted per disruption
        self.dM = dM

        #######################################
        #         INITIAL VARIABLES
        #######################################

        # initial tidal radii
        self.rtau1_i = BBH.r_tau(self.Mstar, self.Rstar, self.Mbh1_i)
        self.rtau2_i = BBH.r_tau(self.Mstar, self.Rstar, self.Mbh2_i)

        # inital probability of a tidal disruption for BH1
        self.p = self.rtau1_i**2/(self.rtau1_i**2 + self.rtau2_i**2)

        # initial angular momentum components of each BH
        # BH1
        self.Jx1_i = 0.
        self.Jy1_i = 0.
        self.Jz1_i = self.spin1_i * (self.G*self.Mbh1_i**2)/self.c

        # BH2
        self.Jx2_i = 0.
        self.Jy2_i = 0.
        self.Jz2_i = self.spin2_i * (self.G*self.Mbh2_i**2)/self.c

        # initial angles of spins of each BH
        # theta is polar angle
        # BH1
        self.theta1_i = 0.
        self.phi1_i = 0.

        # BH2
        self.theta2_i = 0.
        self.phi2_i = 0.

        # initial chi_eff
        self.chi_eff_i = BBH.chi_eff(self.spin1_i, self.spin2_i,
                                    self.Mbh1_i, self.Mbh2_i,
                                    self.theta1_i, self.theta2_i)

        #######################################
        #               LISTS
        #######################################

        # list of number of disruption
        self.N_trials_list = np.arange(0, self.N_trials + 1)

    #In solar units from (Tout et. al 1996)
    def MtoR(self, M):
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

    def run_MC(self, rand_Mstar=False):

        #######################################
        #               LISTS
        #######################################

        # list of spin values for each MC run
        self.MC_spin_BH1_list = []
        self.MC_spin_BH2_list = []

        # list of chi_eff values for each MC run
        self.MC_chi_eff_list = []
        self.MC_chi_eff_f_list = []

        # list of mass values for each MC run
        self.MC_BH1_list = []
        self.MC_BH2_list = []

        # list of mass values for each MC run
        self.MC_Mstar_list = []

        for i in range(self.N_MC):

            #######################################
            #           INITIAL VARIABLES
            #######################################

            # resetting for each MC run
            Mbh1_i = self.Mbh1_i
            Mbh2_i = self.Mbh2_i

            if rand_Mstar == False:
                # these values will range between 0.5 and 10
                Mstar = self.Mstar
                Rstar = self.Rstar
                dM = self.dM

            elif rand_Mstar == True:
                # these values will range between 0.5 and 10
                Mstar = (9.5*np.random.rand() + 0.5) * self.Mo
                Rstar = self.Ro * self.MtoR(Mstar/self.Mo)
                dM = Mstar/2.

            spin1_i = self.spin1_i
            spin2_i = self.spin2_i

            theta1_i = self.theta1_i
            theta2_i = self.theta2_i

            phi1_i = self.phi1_i
            phi2_i = self.phi2_i

            Jx1_i = self.Jx1_i
            Jy1_i = self.Jy1_i
            Jz1_i = self.Jz1_i

            Jx2_i = self.Jx2_i
            Jy2_i = self.Jy2_i
            Jz2_i = self.Jz2_i

            chi_eff = self.chi_eff_i

            p = self.p

            #######################################
            #           LISTS
            #######################################

            # list of masses for a single MC run
            Mbh1_list = [Mbh1_i]
            Mbh2_list = [Mbh2_i]

            # list of spins for a single MC run
            spin1_list = [spin1_i]
            spin2_list = [spin2_i]

            # list of chi_eff for a single MC run
            chi_eff_list = [chi_eff]

            for j in range(self.N_trials):

                #random number between 0 and 1
                trial = np.random.rand()

                # random theta angle between 0 and pi
                rand_theta = np.arccos(2.* np.random.rand() - 1.)

                # random phi angle between 0 and 2pi
                rand_phi = 2. * np.pi * np.random.rand()

                # BH1 disruption
                if trial < p:

                    # get angles
                    theta1_f = rand_theta
                    phi1_f = rand_phi

                    theta2_f = theta2_i
                    phi2_f = phi2_i

                    # cartesian vector components
                    x1 = np.sin(theta1_f) * np.cos(phi1_f)
                    y1 = np.sin(theta1_f) * np.sin(phi1_f)
                    z1 = np.cos(theta1_f)

                    # dot product to determine if current spin and accreted spin are in same direction
                    dp1 = Jx1_i * x1 + Jy1_i * y1 + Jz1_i * z1

                    # efficiency of accretion
                    eta1 = BBH.isco_stuff(spin1_i, Mbh1_i, dp1)[2]

                    dJ1_specific = BBH.isco_stuff(spin1_i, Mbh1_i, dp1)[1]

                    # delta accreted angular momentum
                    dJ1 = dM * dJ1_specific

                    # components of delta accreted angular momentum
                    dJx1 = dJ1 * np.sin(theta1_f) * np.cos(phi1_f)
                    dJy1 = dJ1 * np.sin(theta1_f) * np.sin(phi1_f)
                    dJz1 = dJ1 * np.cos(theta1_f)

                    # adding angular momentum vectorially
                    Jx1_f = Jx1_i + dJx1
                    Jy1_f = Jy1_i + dJy1
                    Jz1_f = Jz1_i + dJz1

                    Jx2_f = Jx2_i
                    Jy2_f = Jy2_i
                    Jz2_f = Jz2_i

                    # angular momentum magnitude
                    J1_mag = np.sqrt(Jx1_f**2 + Jy1_f**2 + Jz1_f**2)

                    # updating mass
                    Mbh1_f = Mbh1_i + dM * eta1
                    Mbh2_f = Mbh2_i

                    # final spin magnitudes from 0 to 1
                    spin1_f = J1_mag * (self.c/(self.G*Mbh1_f**2))
                    spin2_f = spin2_i

                # BH2 disruption
                elif trial > p:

                    # calculating angles
                    theta1_f = theta1_i
                    phi1_f = phi1_i

                    theta2_f = rand_theta
                    phi2_f = rand_phi

                    # cartesian vector components
                    x2 = np.sin(theta2_f) * np.cos(phi2_f)
                    y2 = np.sin(theta2_f) * np.sin(phi2_f)
                    z2 = np.cos(theta2_f)

                    # dot product to determine if current spin and accreted spin are in same direction
                    dp2 = Jx2_i * x2 + Jy2_i * y2 + Jz2_i * z2

                    # efficiency of accretion
                    eta2 = BBH.isco_stuff(spin2_i, Mbh2_i, dp2)[2]

                    dJ2_specific = BBH.isco_stuff(spin2_i, Mbh2_i, dp2)[1]

                    # delta accreted angular momentum
                    dJ2 = dM * dJ2_specific

                    # components of delta accreted angular momentum
                    dJx2 = dJ2 * np.sin(theta2_f) * np.cos(phi2_f)
                    dJy2 = dJ2 * np.sin(theta2_f) * np.sin(phi2_f)
                    dJz2 = dJ2 * np.cos(theta2_f)

                    # adding angular momentum vectorially

                    Jx1_f = Jx1_i
                    Jy1_f = Jy1_i
                    Jz1_f = Jz1_i

                    Jx2_f = Jx2_i + dJx2
                    Jy2_f = Jy2_i + dJy2
                    Jz2_f = Jz2_i + dJz2

                    # angular momentum magnitude
                    J2_mag = np.sqrt(Jx2_f**2 + Jy2_f**2 + Jz2_f**2)

                    # updating mass
                    Mbh1_f = Mbh1_i
                    Mbh2_f = Mbh2_i + dM * eta2

                    # final spin magnitudes from 0 to 1
                    spin1_f = spin1_i
                    spin2_f = J2_mag * (self.c/(self.G*Mbh2_f**2))

                # updating tidal radii
                rtau1_f = BBH.r_tau(Mstar, Rstar, Mbh1_f)
                rtau2_f = BBH.r_tau(Mstar, Rstar, Mbh2_f)

                ## updating probabilities
                p = rtau1_f**2/(rtau1_f**2 + rtau2_f**2)

                # updating chi_eff
                chi_eff = BBH.chi_eff(spin1_f, spin2_f,
                                     Mbh1_f, Mbh2_f,
                                     theta1_f, theta2_f)

                # resetting for loop
                Jx1_i = Jx1_f
                Jy1_i = Jy1_f
                Jz1_i = Jz1_f

                Jx2_i = Jx2_f
                Jy2_i = Jy2_f
                Jz2_i = Jz2_f

                Mbh1_i = Mbh1_f
                Mbh2_i = Mbh2_f

                spin1_i = spin1_f
                spin2_i = spin2_f

                theta1_i = theta1_f
                theta2_i = theta2_f

                phi1_i = phi1_f
                phi2_i = phi2_f

                # appending all quantities
                Mbh1_list.append(Mbh1_f)
                Mbh2_list.append(Mbh2_f)

                spin1_list.append(spin1_f)
                spin2_list.append(spin2_f)

                chi_eff_list.append(chi_eff)

            self.MC_spin_BH1_list.append(spin1_list)
            self.MC_spin_BH2_list.append(spin2_list)
            self.MC_chi_eff_list.append(chi_eff_list)
            self.MC_chi_eff_f_list.append(chi_eff)

            Mbh1_array = np.array(Mbh1_list)
            Mbh2_array = np.array(Mbh2_list)

            self.MC_BH1_list.append(Mbh1_array)
            self.MC_BH2_list.append(Mbh2_array)

            self.MC_Mstar_list.append(Mstar)

            if (i+1)%100 is 0:
                print 'Done with ' + str(i+1) + ' MCs'

        self.MC_mean_chi_eff = np.mean(self.MC_chi_eff_list, axis=0)
        self.MC_std_chi_eff = np.std(self.MC_chi_eff_list, axis=0)

        self.MC_std_chi_eff_pos = self.MC_mean_chi_eff + self.MC_std_chi_eff
        self.MC_std_chi_eff_neg = self.MC_mean_chi_eff - self.MC_std_chi_eff

        print 'Done with this run!\n'
