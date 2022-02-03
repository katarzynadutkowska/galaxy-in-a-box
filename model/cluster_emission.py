#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pylab
import matplotlib.pyplot as plt
import matplotlib
import astropy
from astropy.io import fits
from astropy.convolution import convolve, Gaussian2DKernel
from scipy import stats
from scipy.stats import norm
import sys
import csv
import os
import requests

################################################################################
#
# Constants in CGS units -- will add unit conversion later
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
# Observational fits
#
class ObsFit:

	def __init__(self):
		pass

	def lin_test(self, x, y):
		df = len(x) - 2

		b = np.sum((x-np.mean(x))*(y-np.mean(y))) / np.sum((x-np.mean(x))**2)
		a = np.mean(y)-b*np.mean(x)
		e = y - (a+b*x)
		se_b = (np.sum(e**2) / np.sum((x - np.mean(x))**2))**0.5
		t_b = b / se_b
		p_b = stats.t.sf(np.abs(t_b), df)*2
		# print t_b, stats.t.ppf(0.687, 8)*se_b

		se_a = (np.sum(e**2)/df*(1./df+np.mean(x)**2/np.sum((x-np.mean(x))**2)))**0.5
		t_a = a / se_a
		p_a = stats.t.sf(np.abs(t_a), df)*2
		return np.asarray([a, b, p_a, p_b])


	def correlation_test(self, x, y, tol):
		# Find scaling / correlation between Menv and intensity
		# Return fit parameters and flag indicating which fit is best

		# Always first try linear fit with y = a + b*x:
		fit_m = ofit.lin_test(x, y)
		if (fit_m[3] < tol) & (fit_m[2] > tol):
			return [0, fit_m[1]], 'lin'
		elif (fit_m[3] < tol) & (fit_m[2] < tol) & (fit_m[0] > 0):
			return [fit_m[0], fit_m[1]], 'lin'
		# If a linear fit is not good enough, try power-law:
		else:
			fit_m = ofit.lin_test(np.log10(x), np.log10(y))
			if (fit_m[3] < tol) & (fit_m[2] > tol):
				return [0, fit_m[1]], 'pow'
			elif (fit_m[3] < tol) & (fit_m[2] < tol):
				return [fit_m[0], fit_m[1]], 'pow'
			else:
				sys.exit()



ofit = ObsFit()

################################################################################
#
# Loading data from MySQL Water_Emission_Database
#
colnames = ['obs_id', 'object', 'obj_type', 'ra_2000', 'dec_2000', 'transition', 'freq',\
            'telescope', 'instrument', 'obs_res', 'distance', 'luminosity', 'tbol', 'menv',\
            'vlsr', 'flux', 'flux_err', 'unit','ref','extra']

url = 'https://katarzynadutkowska.github.io/WED/Database/WED_988.csv'
flux = []
lbol = []
diss = []
menv = []
beam = []

with requests.Session() as s:
    download = s.get(url)
    decoded_content = download.content.decode('utf-8')
    data = csv.reader(decoded_content.splitlines(), delimiter=',')
    next(data)
    data_list = list(data)
    for row in data_list:
        flux.append(row[colnames.index('flux')])
        lbol.append(row[colnames.index('luminosity')])
        diss.append(row[colnames.index('distance')])
        menv.append(row[colnames.index('menv')])
        beam.append(row[colnames.index('obs_res')])

flux = [i == 'NaN' if i == 'NULL' else float(i) for i in flux]
lbol = [i == 'NaN' if i == 'NULL' else float(i) for i in lbol]
diss = [i == 'NaN' if i == 'NULL' else float(i) for i in diss]
menv = [i == 'NaN' if i == 'NULL' else float(i) for i in menv]
beam = [i == 'NaN' if i == 'NULL' else float(i) for i in beam]

TdV  = []
Dist = []
Menv = []
obs_res = []
l_bol = []

for i in range(len(flux)):
    if flux[i] != None:
        if flux[i] >= 0 and menv[i] != None:
            TdV.append(flux[i])
            l_bol.append(lbol[i])
            Dist.append(diss[i])
            Menv.append(menv[i])
            obs_res.append(beam[i])

TdV   = np.asarray(TdV)
l_bol = np.asarray(l_bol)
Dist  = np.asarray(Dist)
Menv  = np.asarray(Menv)
obs_res = np.asarray(obs_res)

################################################################################
#
# Basic 1D template model to prove concept (to implement: spectral cubes)
#
class Mod_Template:

	############################################################################
	# Random mass plus radial distributions
	############################################################################
    def main(self,output = 1, FILE = "./results/galaxycluster_emission.csv"):

        config={}
        f=open('./setup_files/image_setup_change.dat','r')
        for line in f.readlines():
            config[line.split()[0]]=line.split()[1]
        # Parameters relating to new image
        dist = float(config['bob']) # distance to cluster in pc
        classI_scale = 0.1
        factor = 485.5 # for h2o  at 988 ghz
        #factor = 574.5 # for co at 1152 ghz
        model = np.load(config['dist'])

        tol_file = np.int32(config['tol'])   # fit tolerance: if probability greater, then hypothesis is rejected at 1 sigma
        mean_val = 0                         # mean of the distribution
        std_val  = 1                         # standard deviation of the distribution
        tol      = norm.cdf(tol_file*std_val,mean_val,std_val) - norm.cdf(-tol_file*std_val,mean_val,std_val) # calculates sigma following the Empirical Rule

        menv = Menv
        i_dist = TdV*(Dist/dist)**2*factor
        fit, flag = ofit.correlation_test(menv, i_dist, tol)

        # Isolate Class 0 and I sources from the model
        cl0    = (model[6] == 10)
        cl1    = (model[6] == 11)
        cl0_hm = (model[6] == 3)
        cl1_hm = (model[6] == 4)
        cl_hm  = (model[6] == 2)

        if flag == 'lin':
            cl0int = fit[0] + fit[1]*model[2][cl0]
            cl1int = classI_scale * (fit[0] + fit[1]*model[2][cl1])
            cl0int_hm = fit[0] + fit[1]*model[2][cl0_hm]
            cl1int_hm = classI_scale * (fit[0] + fit[1]*model[2][cl1_hm])
        elif flag == 'pow':
            cl0int = fit[0]*model[2][cl0]**fit[1]
            cl1int = classI_scale * (fit[0]*model[2][cl1]**fit[1])
            cl0int_hm = fit[0]*model[2][cl1_hm]**fit[1]
            cl1int_hm = classI_scale * (fit[0]*model[2][cl1_hm]**fit[1])

        im = [sum(cl0int)+sum(cl1int)+sum(cl0int_hm)+sum(cl1int_hm)]
        mass=[sum(model[2])/0.03]
        N=[len(model[2])]
        if output == 1:
            print('Total emission from cluster is '+str(im))
            print('Total mass in cluster is '+str(mass))

        config={}
        for line in open("./setup_files/cluster_setup_change.dat","r").readlines():
            config[line.split()[0]]=float(line.split()[1])

        imf_type = config['imf']
        tffscale = config['tff']
        SFE = config['SFE']

        filename = FILE

        if os.path.exists(filename):
            append_write = 'a' # append if already exist
        else:
            append_write = 'w' # make a new file if not

        with open(filename, append_write) as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerows(zip(im,mass,N))


################################################################################
#
# Main
#
if __name__ == "__main__":
    template=Mod_Template()
    template.main()
