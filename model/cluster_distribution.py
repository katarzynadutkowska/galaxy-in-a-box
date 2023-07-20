#!/usr/bin/env python
# coding: utf-8

import numpy as np
import numpy.random
from math import *
import pylab
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
import sys
from sympy.solvers import nsolve
from sympy import Symbol, exp
import os

path_main             = os.getcwd()
results_path          = os.path.join(path_main,"results")
setup_files_cluster   = os.path.join(path_main,"setup_files","cluster")
cluster_SETUP         = os.path.join(setup_files_cluster,"cluster_setup_change.dat")
dist_file             = os.path.join(results_path,"distribution.npy")

################################################################################
#
# Various functions used
#
class Mod_MyFunctions:

    def __init__(self):
        pass

    def imf(self, x, imf_type):

        # Chabrier (2003) IMF for young clusters plus disk stars: lognorm and power-law tail
        mnorm = 1.0
        A1 = 0.158
        mc = 0.079
        sigma = 0.69
        A2 = 4.43e-2
        x0 = -1.3

        if imf_type == 0:
            a1 = A1 * np.exp(-((x - np.log10(mc))**2)/2.0/sigma**2)
            a2 = A2 * (10.0**x)**(x0-1)
            return np.where(x <= np.log10(mnorm), a1, a2)

        if imf_type == 1:
            a1 = A1 * np.exp(-((x - np.log10(mc))**2)/2.0/sigma**2)
            a2 = A2 * (10.0**x)**(x0-0)
            return np.where(x <= np.log10(mnorm), a1, a2)

        if imf_type == 2:
            a1 = A1 * np.exp(-((x - np.log10(mc))**2)/2.0/sigma**2)
            a2 = A2 * (10.0**x)**(x0-2)
            return np.where(x <= np.log10(mnorm), a1, a2)


    def mass_dist(self,
        mmin=0.01,
        mmax=100,
        Mcm=10000,
        imf_type=0,
        SFE=0.03):
        mmin_log = np.log10(mmin)
        mmax_log = np.log10(mmax)

        chunksize = int(Mcm * 0.5)
        result = np.array([], dtype=np.float64)
        while result.sum() <= SFE * Mcm:
            x = np.random.uniform(mmin_log, mmax_log, size=chunksize)
            y = np.random.uniform(0, 1, size=chunksize)
            result = np.hstack((result, 10 ** x[y < myf.imf(x, imf_type)]))
        return result[result.cumsum() <= SFE * Mcm]

    def age_dist(self, m_temp, r, sfr, tffscale = 1.0):
        density=3*sum(m_temp)/(4*np.pi*r**3)
        G=4.43*10**(-3) #pc^3 Msol^-1 Myr^-2
        tff=0.5427*1/np.sqrt(G*density) #free fall time for cluster

        tff_randomized = tff*np.random.uniform(0.0,1.0,size=1)

        if sfr != 0.0:
            tffscale = len(m_temp) * np.average(m_temp) * tff_randomized**(-1) * sfr**(-1)
        else:
            pass

        age=tffscale*tff_randomized

        lmbdaSF=1/age

        lmbda=numpy.zeros(5)

        ### Relative population fractions observed in clouds where t = 1.2 Myr

        lmbda[0]=14.7
        lmbda[1]=7.9
        lmbda[2]=8.0
        lmbda[3]=0.347

        #Calculate new fractional populations, setting Class III equal to any leftovers

        self.frac = numpy.zeros(5)

        self.frac[0] = (lmbdaSF/lmbda[0])*(1-np.exp(-lmbda[0]*age))

        self.frac[1] = (lmbdaSF/lmbda[1])*(1-(lmbda[1]/(lmbda[1]-lmbda[0]))*(np.exp(-lmbda[0]*age))-(lmbda[0]/(lmbda[0]-lmbda[1]))*np.exp(-lmbda[1]*age))

        self.frac[2] = (lmbdaSF/lmbda[2])*(1-(lmbda[1]*lmbda[2])/((lmbda[2]-lmbda[0])*(lmbda[1]-lmbda[0]))*(np.exp(-lmbda[0]*age))-
            (lmbda[0]*lmbda[2])/((lmbda[2]-lmbda[1])*(lmbda[0]-lmbda[1]))*(np.exp(-lmbda[1]*age))-
            (lmbda[0]*lmbda[1])/((lmbda[0]-lmbda[2])*(lmbda[1]-lmbda[2]))*(np.exp(-lmbda[2]*age)))

        self.frac[3] = (lmbdaSF/lmbda[3])*(1-(lmbda[1]*lmbda[2]*lmbda[3])/((lmbda[1]-lmbda[0])*(lmbda[2]-lmbda[0])*(lmbda[3]-lmbda[0]))*(np.exp(-lmbda[0]*age)) -
            (lmbda[0]*lmbda[2]*lmbda[3])/((lmbda[0]-lmbda[1])*(lmbda[2]-lmbda[1])*(lmbda[3]-lmbda[1]))*(np.exp(-lmbda[1]*age)) -
            (lmbda[0]*lmbda[1]*lmbda[3])/((lmbda[0]-lmbda[2])*(lmbda[1]-lmbda[2])*(lmbda[3]-lmbda[2]))*(np.exp(-lmbda[2]*age)) -
            (lmbda[0]*lmbda[1]*lmbda[2])/((lmbda[0]-lmbda[3])*(lmbda[1]-lmbda[3])*(lmbda[2]-lmbda[3]))*(np.exp(-lmbda[3]*age)))

        # Assume that everything ends up as Class III; no main sequence. An ok assumption when the interest is on Class 0/I sources

        self.frac[4] = 1. - sum(self.frac[:4])

        return self.frac, age

myf=Mod_MyFunctions()

#
# Constants in CGS units
#
class CgsConst:

    def __init__(self):
        self.c = 2.99792458e10    # cm s**-1
        self.k = 1.3806504e-16    # erg K**-1
        self.h = 6.62606896e-27   # erg s**-1
        self.AMU = 1.660538e-24   # g
        self.sigma = 5.670400e-5  # erg cm**-2 s**-1 K**-4
        self.hck = self.h*self.c/self.k
        self.yr = 365.2422*24.*3600.
        self.pcau = 206265.

const=CgsConst()

################################################################################
#
# Module for mass, radial distributions
#

class Mod_MassRad:

    ############################################################################
    # Random mass plus radial distributions
    ############################################################################
    def __init__(self):
        pass

    def mass_radius(self,
        Mcm=10000,
        N0=300,
        r0=1.0,
        alpha=0.33,
        p=1.0,
        dv=2.0,
        tffscale=1.0,
        SFE=0.03,
        imf_type=0,
        sfr = 0.0,
        output=1,
        FILE_dist = dist_file):

        m_temp = myf.mass_dist(mmin = 0.01, mmax = 100., Mcm = Mcm, imf_type = imf_type, SFE=SFE)
        N=len(m_temp)

        self.m = m_temp
        self.N = N

        # Spatial distribution
        r = r0 * (N/N0)**alpha
        rad = r*numpy.random.power(2.-p, size=(N))
        rad_m = rad*(self.m/min(self.m))**(-0.15)
        phi = numpy.random.rand(N)*2*pi

        self.x = numpy.zeros(N)
        self.y = numpy.zeros(N)

        # 2. Calculate age distribution

        age_temp, age= myf.age_dist(m_temp, r, tffscale = tffscale, sfr = sfr)

        #print ('Age fractions calculated')

        # 3. Sort out BD, LM and HM stars; for this project, ignore BD and HM
        hm = numpy.asarray((m_temp > 10.).nonzero())[0]
        lm = numpy.asarray(((m_temp <= 10.) & (m_temp > 0.05)).nonzero())[0]
        bd = numpy.asarray((m_temp <= 0.05).nonzero())[0]

        nC0 = int(numpy.round(numpy.size(lm)*age_temp[0]))
        nCI = int(numpy.round(numpy.size(lm)*age_temp[1]))

        nC0_hm = int(numpy.round(numpy.size(hm)*age_temp[0]))
        nCI_hm = int(numpy.round(numpy.size(hm)*age_temp[1]))

        lm0 = lm[:nC0]
        lmi = lm[nC0:nC0+nCI]
        lmii = lm[nC0+nCI:]

        hm0 = hm[:nC0_hm]
        hmi = hm[nC0_hm:nC0_hm+nCI_hm]
        hmii = hm[nC0_hm+nCI_hm:]

        # 4. If Class 0 source, assume envelope mass is 3 times higher; for Class I's, M_env is 1.5 times higher

        self.m[lm0] = 3. * m_temp[lm0]
        self.m[lmi] = 1.5 * m_temp[lmi]
        self.m[hm0] = 3. * m_temp[hm0]
        self.m[hmi] = 1.5 * m_temp[hmi]

        self.mass_flag = numpy.zeros(N)
        self.mass_flag[hm] = 2
        self.mass_flag[hm0] = 3
        self.mass_flag[hmi] = 4
        self.mass_flag[hmii] = 5
        self.mass_flag[lm0] = 10
        self.mass_flag[lmi] = 11
        self.mass_flag[lmii] = 12
        self.mass_flag[bd] = 0

        #if not os.path.exists('output_distribution'):
        if output == 1:
            f=open(os.path.join(results_path,"output_distribution"),'w')
            f.write('min(M), max(M) = %4.2f, %4.2f Msun\n' %(min(m_temp), max(m_temp)))
            f.write('Total cluster mass: %6.2f \n' %(sum(self.m)))
            f.write('Rmax = %4.2f pc \n' %r)
            f.write('Age distribution (Class 0, I, Flat, II, III): %4.2f, %4.2f, %4.2f, %4.2f, %4.2f \n' %(age_temp[0], age_temp[1], age_temp[2], age_temp[3], age_temp[4]))
            f.write(' \n')
            f.write('Number of HM cores: %3i \n' %(numpy.size(hm)))
            f.write('Number of LM cores: %3i \n' %(numpy.size(lm)))
            f.write('Number of BD cores: %3i \n' %(numpy.size(bd)))
            f.write(' \n')
            f.write('Number of LM Class 0 sources: %3i \n' %(nC0))
            f.write('Number of LM Class I sources: %3i \n' %(nCI))
            f.write('Number of HM Class 0 sources: %3i \n' %(nC0_hm))
            f.write('Number of HM Class I sources: %3i \n' %(nCI_hm))
            f.close()

        x = rad_m[:]*np.cos(phi[:])
        y= rad_m[:]*np.sin(phi[:])

        # Outflow inclination, PA and protostellar velocity dispersion
        self.i = numpy.random.rand(N)*90.
        self.pa = numpy.random.rand(N)*180.
        self.vel = numpy.random.normal(dv, size=N)

        #print ('Spatial distribution calculated')
        sfr_cluster = self.N*np.average(self.m)*(age*1e6)**(-1)
        space_dist=x,y,self.m,self.i,self.pa,self.vel,self.mass_flag
        np.save(FILE_dist,space_dist)
        np.save("sfr_temp.npy",sfr_cluster)

        





################################################################################
#
# Main Module for distribution
#
class Mod_distribution:

    ############################################################################
    # define configuration etc
    ############################################################################
    def __init__(self, FILE_cluster = cluster_SETUP):

        ########################################################################
        ########################################################################
        ###                                                                  ###
        ###       EDIT THIS PART ONLY                                        ###
        ###                                                                  ###
        ########################################################################
        ########################################################################

        config={}
        for line in open(FILE_cluster,"r").readlines():
            config[line.split()[0]]=float(line.split()[1])
            # print "%20s=%.5e" % (line.split()[0],config[line.split()[0]])

        ### Cluster parameters
        self.Mcm = int(config['Mcm'])   # Molecular Cloud Mass

        ### Initial mass function (currently only Chabrier 2003 IMF available)
        self.imf_type = config['imf']

        ### Free fall time scaling and star formation efficiency
        self.tffscale = config['tff']
        self.SFE = config['SFE']

        ### Radial distribution of stars, from Adams et al. (2014)
        self.r0 = config['r0']   # initial radius (pc)
        self.N0 = int(config['N0'])   # initial number of stars
        self.alpha = config['alpha']   # power-law index for maximum cluster radius
        self.p = config['p']   # power-law index for radius PDF

        self.dv = config['dv']   # internal velocity dispersion (only relevant if creating spectral cubes; not implemented at the moment)

        self.sfr = config['sfr']

        self.massrad=Mod_MassRad()


    ############################################################################
    # begin calculation
    ############################################################################
    def calc(self, output = 1, FILE_dist = dist_file):

            self.massrad.mass_radius(
            Mcm = self.Mcm,
            N0 = self.N0,
            r0 = self.r0,
            alpha = self.alpha,
            p = self.p,
            dv = self.dv,
            tffscale = self.tffscale,
            SFE = self.SFE,
            imf_type = self.imf_type,
            output= output,
            FILE_dist = FILE_dist)


            mass = self.massrad.m
            N = self.massrad.N
            if output == 1:
                print("Number of stars in cluster is "+str(N))
                print("Average protostellar mass is "+str(np.average(mass))+" M_sun")

################################################################################
#
#Main



if __name__ == "__main__":
    distribution=Mod_distribution()
    distribution.calc()
