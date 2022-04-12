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
        
        def ThreeBreakIMF(x, A1, A2, A3, x1, x2, x3, k0, k1, k2, k3):
            #x1 = x[(x > np.log10(k0) ) & (x <= np.log10(k1) )]
            #x2 = x[(x > np.log10(k1) ) & (x <= np.log10(k2) )]
            #x3 = x[(x > np.log10(k2) ) & (x <= np.log10(k3) )]
#            x4 = x[(x > np.log10(k3) ) & (x <= np.log10(k4) )]

            #a1 = A1 * (10.0**x1)**(-x1)
            #a2 = A2 * (10.0**x2)**(-x2)
            #a3 = A3 * (10.0**x3)**(-x3)
#            a4 = A4 * (10.0**x4)**(-x4)

            y=[]#np.zeros(len(x))    
            for i in x:
                if (i > np.log10(k0)) and (i <= np.log10(k1)):
                    y.append(A1*(10.0**i)**(-x1))
                if (i > np.log10(k1)) and (i <= np.log10(k2)):
                    y.append(A2*(10.0**i)**(-x2))
                if (i > np.log10(k2)) and (i <= np.log10(k3)):
                    y.append(A3*(10.0**i)**(-x3))

            #a1 = A1 *(10.0**x)**(-x1)
            #a2 = A2 *(10.0**x)**(-x2)
            #a3 = A3 *(10.0**x)**(-x3)
            #aa1_ = np.where((x > np.log10(k0)) & (x <= np.log10(k1)), a1, 5); aa1 = aa1_[aa1_ != 5]
            #aa2_ = np.where((x > np.log10(k1)) & (x <= np.log10(k2)), a2, 5); aa2 = aa2_[aa2_ != 5]
            #aa3_ = np.where((x > np.log10(k2)) & (x <= np.log10(k3)), a3, 5); aa3 = aa3_[aa3_ != 5]
            #Arr3B = np.concatenate((aa1,aa2,aa3))

            #a1 = A1 * (10.0**x[(x > np.log10(k0) ) & (x <= np.log10(k1) )])**(-x1)
            #a2 = A2 * (10.0**x[(x > np.log10(k1) ) & (x <= np.log10(k2) )])**(-x2)
            #a3 = A3 * (10.0**x[(x > np.log10(k2) ) & (x <= np.log10(k3) )])**(-x3)
            ##a4 = A4 * (10.0**x[(x > np.log10(k3)  ) & (x <= np.log10(k4) )])**(-x4)


            #a1 = A1 *(10.0**x[np.where((x > np.log10(k0)) & (x <= np.log10(k1)))])**(-x1)
            #a2 = A2 *(10.0**x[np.where((x > np.log10(k1)) & (x <= np.log10(k2)))])**(-x2)
            #a3 = A3 *(10.0**x[np.where((x > np.log10(k2)) & (x <= np.log10(k3)))])**(-x3)
            #Arr3B = np.concatenate((a1,a2,a3))
            return y # Arr3B


        def FourBreakIMF(x, A1, A2, A3, A4, x1, x2, x3, x4, k0, k1, k2, k3, k4):
            y=[]#np.zeros(len(x))    
            for i in x:
                if (i > np.log10(k0)) and (i <= np.log10(k1)):
                   y.append(A1*(10.0**i)**(-x1))
                if (i > np.log10(k1)) and (i <= np.log10(k2)):
                    y.append(A2*(10.0**i)**(-x2))
                if (i > np.log10(k2)) and (i <= np.log10(k3)):
                    y.append(A3*(10.0**i)**(-x3))
                if (i > np.log10(k3)) and (i <= np.log10(k4)):
                    y.append(A4*(10.0**i)**(-x4))
            #a1 = A1 *(10.0**x[np.where((x > np.log10(k0)) & (x <= np.log10(k1)))])**(-x1)
            #a2 = A2 *(10.0**x[np.where((x > np.log10(k1)) & (x <= np.log10(k2)))])**(-x2)
            #a3 = A3 *(10.0**x[np.where((x > np.log10(k2)) & (x <= np.log10(k3)))])**(-x3)
            #a4 = A4 *(10.0**x[np.where((x > np.log10(k3)) & (x <= np.log10(k4)))])**(-x4)
            #Arr4B = np.concatenate((a1,a2,a3,a4))
            return y

        # Chabrier 2001, from 0.1 to 100
        if imf_type == 3:
            
            A1Chabrier01 = 0.376613899390368
            
            return A1Chabrier01 * 40.33 * (10.0**x)**(-3.3) * np.exp(- (716.4/(10.0**x))**(0.25) )
        
        # Chabrier 2005, from 0.1 to 100
        if imf_type == 4:
            
            A1Chabrier05 = 0.9652005
            mcChabrier05 = 0.2
            sigmaChabrier05 = 0.55
            A2Chabrier05 = 0.4255185
            x0Chabrier05 = 2.35

            a1 = A1Chabrier05 * np.exp(-((x-np.log10(mcChabrier05))**2)/(2*sigmaChabrier05**2))
            a2 = A2Chabrier05 * (10.0**x)**(-x0Chabrier05)
            
            return np.where(x <= np.log10(mnorm), a1, a2)

        # Cannonical IMF, from 0.1 to 100
        if imf_type == 5:
            
            A1Caonical = 0.289159021998465
            A2Caonical = 0.139654602092348
            
            a1 = A1Caonical * (10.0**x)**(-1.3)
            a2 = A2Caonical * (10.0**x)**(-2.35)
            return np.where(x <= np.log10(0.5), a1, a2)
        
        # Salpeter 1955, Bottom-heavy, from 0.01 to 100 (originally defied from exp(-0.4) to exp(1))
        if imf_type == 6:

            A1Salpeter = 0.926709878283741
            a1 = A1Salpeter * (10.0**x)**(-1.3)
            
            return a1

        # Kennicutt (1983), Universal, from 0.1 to 100
        if imf_type == 7:

            AKennicutt = 0.224935641926054
            
            a1 = AKennicutt * (10.0**x)**(-1.4)
            a2 = AKennicutt * (10.0**x)**(-2.5)
            return np.where(x <= np.log10(mnorm), a1, a2)
        
        # Hopkins & Beacom (2006), Top-heavy, from 0.1 to 100
        if imf_type == 8:

            A1HnB06 = 0.211725271146086
            A2HnB06 = 0.134928347205648
            
            a1 = A1HnB06 * (10.0**x)**(-1.5)
            a2 = A2HnB06 * (10.0**x)**(-2.15)
            return np.where(x <= np.log10(0.5), a1, a2)
        
        # Davé (2008), Top-heavy, from 0.1 to 100
        if imf_type == 9:

            A1Dave08 = 0.286276391369702
            A2Dave08 = 0.143138195684851
            
            a1 = A1Dave08 * (10.0**x)**(-1.3)
            a2 = A2Dave08 * (10.0**x)**(-2.3)
            return np.where(x <= np.log10(0.5), a1, a2)
        
        # Hoversten & Glazebrook (2008), Universal, from 0.1 to 100
        if imf_type == 10:

            A1HnG08 = 0.223780768686911
            A2HnG08 = 0.115555510479912
            
            a1 = A1HnG08 * (10.0**x)**(-1.5)
            a2 = A2HnG08 * (10.0**x)**(-2.4535)
            return np.where(x <= np.log10(0.5), a1, a2)

        # Baldry & Glazebrook (2003), Universal, from 0.1 to 100
        if imf_type == 11:

            A1HnG08 = 0.214013734931171
            A2HnG08 = 0.131740907069795
            
            a1 = A1HnG08 * (10.0**x)**(-1.5)
            a2 = A2HnG08 * (10.0**x)**(-2.2)
            return np.where(x <= np.log10(0.5), a1, a2)

        # Scalo 1986 (Fit), Universal, from 0.1 to 100
        if imf_type == 12:
            
            A1Scalo86 = 0.355727158636779
            A2Scalo86 = 0.184135924897207
            A3Scalo86 = 0.184135924897207
            A4Scalo86 = 0.0518965547762883

            Rezultz = FourBreakIMF(x,A1=A1Scalo86,A2=A2Scalo86,A3=A3Scalo86,A4=A4Scalo86,x1=1.15,x2=2.1,x3=3.05,x4=2.5,k0=0.1,k1=0.5,k2=1,k3=10,k4=100)

            return Rezultz

        # Scalo 1998, Universal, from 0.1 to 100
        if imf_type == 13:
            
            A1Scalo98 = 0.284451049966366
            A2Scalo98 = 0.284451049966366
            A3Scalo98 = 0.113242002663081

            Rezultz = ThreeBreakIMF(x,A1=A1Scalo98,A2=A2Scalo98,A3=A3Scalo98,x1=1.15,x2=2.1,x3=3.05,k0=0.1,k1=1,k2=10,k3=100)

            return Rezultz

        # van Dokkum & Conroy Very Bottom Heavy 2010, Bottom-heavy, from 0.1 to 100
        if imf_type == 14:
            
            A1vanDnCVBH = 0.00790569440042097

            a1 = A1vanDnCVBH * (10.0**x)**(-3.5)

            return a1
        
        # van Dokkum & Conroy Bottom Heavy 2010, Bottom-heavy, from 0.1 to 100
        if imf_type == 15:
            
            A1vanDnCBH = 0.0200000200000200

            a1 = A1vanDnCBH * (10.0**x)**(-3.0)

            return a1

        # Renzini Steep IMF 2005, Bottom-heavy, from 0.1 to 100
        if imf_type == 16:
            
            A1steepR05 = 0.0104970653510996

            a1 = A1steepR05 * (10.0**x)**(-3.35)

            return a1

        # Renzini Flat IMF 2005, Top-heavy, from 0.1 to 100
        if imf_type == 17:
            
            A1flatR05 = 0.171636364325098

            a1 = A1flatR05 * (10.0**x)**(-1.35)

            return a1

        # Fardal et al. Paunchy 2007, Middle-heavy, from 0.1 to 100
        if imf_type == 18:
            
            A1FardalPaunchy07 = 0.350911129985083
            A2FardalPaunchy07 = 0.216011138630843
            A3FardalPaunchy07 = 0.752194473653271

            Rezultz = ThreeBreakIMF(x,A1=A1FardalPaunchy07,A2=A2FardalPaunchy07,A3=A3FardalPaunchy07,x1=1,x2=1.7,x3=2.6,k0=0.1,k1=0.5,k2=4,k3=100)

            return Rezultz

        # Fardal et al. Extremely Top Heavy 2007, Top-heavy, from 0.1 to 100
        if imf_type == 19:
            
            A1ETHFardal07 = 0.106742530991320

            a1 = A1ETHFardal07 * (10.0**x)**(-1.95)

            return a1

        # Kroupa 2001, Universal, from 0.01 to 100
        if imf_type == 20:
            
            A1Kroupa01 = 0.350911129985083
            A2Kroupa01 = 0.158972099575776
            A3Kroupa01 = 0.0794860497878882

            Rezultz = ThreeBreakIMF(x,A1=A1Kroupa01,A2=A2Kroupa01,A3=A3Kroupa01,x1=0.3,x2=1.3,x3=2.3,k0=0.01,k1=0.08,k2=0.5,k3=100)

            return Rezultz
        
        # Modified Kroupa01 1 (form Meurer et al. 2009), Top-light, from 0.01 to 100
        if imf_type == 21:
            
            A1mod1Kroupa01 = 2.12598723577527
            A2mod1Kroupa01 = 0.170078978862021
            A3mod1Kroupa01 = 0.0425197447155053

            Rezultz = ThreeBreakIMF(x,A1=A1mod1Kroupa01,A2=A2mod1Kroupa01,A3=A3mod1Kroupa01,x1=0.3,x2=1.3,x3=3.3,k0=0.01,k1=0.08,k2=0.5,k3=100)

            return Rezultz

        # Modified Kroupa01 2 (form Meurer et al. 2009), Top-heavy, from 0.01 to 100
        if imf_type == 22:
            
            A1mod2Kroupa01 = 1.45165470227029
            A2mod2Kroupa01 = 0.116132376181623

            a1 = A1mod2Kroupa01 * (10.0**x)**(-0.3)
            a2 = A2mod2Kroupa01 * (10.0**x)**(-1.3)

            return np.where(x <= np.log10(0.08), a1, a2)


        # Modified Kroupa01 2 (form Meurer et al. 2009), Top-heavy, from 0.01 to 100
        if imf_type == 220:
            
            A1mod2Kroupa01 = 1.45165470227029
            A2mod2Kroupa01 = 0.116132376181623

            #a1 = A1mod2Kroupa01 * (10.0**x)**(-0.3)
            #a2 = A2mod2Kroupa01 * (10.0**x)**(-1.3)

            Rezultz = ThreeBreakIMF(x,A1=A1mod2Kroupa01,A2=A2mod2Kroupa01,A3=A2mod2Kroupa01,x1=0.3,x2=1.3,x3=1.3,k0=0.01,k1=0.08,k2=0.5,k3=100)

            return Rezultz

        # Modified Kroupa01 3 (form Wilkins et al. 2008b), Universal, from 0.01 to 100
        if imf_type == 23:
            
            A1mod3Kroupa01 = 1.99821313724425
            A2mod3Kroupa01 = 0.159857050979540
            A3mod3Kroupa01 = 0.0772058664879645

            Rezultz = ThreeBreakIMF(x,A1=A1mod3Kroupa01,A2=A2mod3Kroupa01,A3=A3mod3Kroupa01,x1=0.3,x2=1.3,x3=2.35,k0=0.01,k1=0.08,k2=0.5,k3=100)

            return Rezultz

        # van Dokkum 2008, Bottom-light, from 0.01 to 100
        if imf_type == 24:
            
            #A1Chabrier05 = 0.9652005
            #mcChabrier05 = 0.2
            #sigmaChabrier05 = 0.55
            #A2Chabrier05 = 0.4255185
            #x0Chabrier05 = 2.35

            A1VD08             = 0.511
            mcVD08             = 0.079 # 0.079, 0.4, 2
            sigVD08            = 0.69 # 0.69
            AhVD08             = 0.32
            xVD08              = 2.3
            ncVD08             = 25

            a1 = A1VD08*(0.5*ncVD08*mcVD08)**(-xVD08) * np.exp(-((np.log10(x) - np.log10(mcVD08))**2)/(2*sigVD08**2))
            a2 = AhVD08 * (10.0**x)**(-xVD08)
            
            return np.where(x <= np.log10(ncVD08*mcVD08), a1, a2)

        # Larson 1998 (from Portinari 2004), Bottom-light, from 0.01 to 100
        if imf_type == 25:
            
            A1Larson04 = 0.817193205330865
            
            return A1Larson04 * 0.317 * (10.0**x)**(-2.35) * np.exp(- 0.3375/(10.0**x))

        # Modified Larson (from Portinari 2004), Bottom-light, from 0.01 to 100
        if imf_type == 26:
            
            A1Larson04 = 0.592524862490177
            
            return A1Larson04 * 0.4337 * (10.0**x)**(-2.7) * np.exp(- 0.425/(10.0**x))

        # Cappellari et al. BH 2012, Bottom-heavy, from 0.01 to 100
        if imf_type == 27:
            
            A1BHCappellari12 = 0.000452139586199803

            a1 = A1BHCappellari12 * (10.0**x)**(-2.8)

            return a1

        # Cappellari et al. TH 2012, Top-heavy, from 0.01 to 100
        if imf_type == 28:
            
            A1THCappellari12 = 0.0505050505050505

            a1 = A1THCappellari12 * (10.0**x)**(-1.5)

            return a1

        # Miller-Scalo 1979, Universal, from 0.1 to 100
        if imf_type == 29:
            
            #y=[]#np.zeros(len(x))    
            #for i in x:
            #    if (i > np.log10(0.01)) and (i <= np.log10(1)):
            #        y.append(0.225276926316436*(10.0**i)**(-1.4))
            #    if (i > np.log10(1)) and (i <= np.log10(10)):
            #        y.append(0.225276926316436*(10.0**i)**(-2.5))
            #    if (i > np.log10(10)) and (i <= np.log10(100)):
            #        y.append(1.42140131201279*(10.0**i)**(-3.3))

            A1MS1979 = 0.225276926316436
            A2MS1979 = 0.225276926316436
            A3MS1979 = 1.42140131201279

            Rezultz = ThreeBreakIMF(x,A1=A1MS1979,A2=A2MS1979,A3=A3MS1979,x1=1.4,x2=2.5,x3=3.3,k0=0.1,k1=1,k2=10,k3=100)

            return Rezultz
        
        if imf_type == 30:
            
            return (10.0**x)**(-1) #np.where(x<= np.log10(1), (10.0**x)**1, (10.0**x)**2)
        #


    def mass_dist(self,
        mmin=0.01, # mmin=0.01,
        mmax=100,
        Mcm=10000,
        imf_type=0,
        SFE=0.03):
        mmin_log = np.log10(mmin)
        mmax_log = np.log10(mmax)

        chunksize = int(Mcm * 0.6)
        result = np.array([], dtype=np.float64)
        while result.sum() <= SFE * Mcm:
            x = np.random.uniform(mmin_log, mmax_log, size=chunksize)
            y = np.random.uniform(0, 10, size=chunksize)
            result = np.hstack((result, 10 ** x[y < myf.imf(x, imf_type)]))
        return result[result.cumsum() <= SFE * Mcm]

    def age_dist(self, m_temp, r, tffscale = 1.0):
        density=3*sum(m_temp)/(4*np.pi*r**3)
        G=4.43*10**(-3) ##pc^3 Msol^-1 Myr^-2
        tff=0.5427*1/np.sqrt(G*density) #free fall time for cluster

        age=tffscale*tff*np.random.uniform(0.0,1.0,size=1) #calculating current age from free fall time tff

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

        return self.frac

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

        age_temp= myf.age_dist(m_temp, r, tffscale = tffscale)

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

        space_dist=x,y,self.m,self.i,self.pa,self.vel,self.mass_flag
        np.save(FILE_dist,space_dist)




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
            # temp = numpy.zeros(N)
            # temp2 = numpy.zeros(20)
            # for i in range(0,N): temp[i] = log10(mass[i])

################################################################################
#
#Main



if __name__ == "__main__":
    distribution=Mod_distribution()
    distribution.calc()
