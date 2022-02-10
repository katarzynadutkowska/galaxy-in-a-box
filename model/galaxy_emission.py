# Galaxy-in-a-box authors: Katarzyna M. Dutkowska, Lars E. Kristensen,
# with contribution of Thomas Jones and Markus Rasmussen
# Copenhagen University, NBI & StarPlan, Copenhagen, Denmark
# 2020-2021

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import matplotlib
import math
import os
import sys
from astropy.io import fits
from matplotlib.colors import LogNorm
import copy
import csv
from astropy.convolution import Gaussian2DKernel
from astropy.convolution import convolve
from matplotlib import patches
from mpl_toolkits.axes_grid1.anchored_artists import (AnchoredOffsetbox, AuxTransformBox)
from matplotlib.patches import Ellipse
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from timer import Timer # Here we call the 'timer.py' script added in the folder
import cluster_distribution as cd
import cluster_emission as ce
import time
import shutil
timestr = time.strftime("%Y%m%d-%H%M%S") # Here we save the time of running the
                                         # script; can be used later to save files

################################################################################
######## Defining paths to main folders: main, results and setup files #########
################################################################################

path_main     = os.getcwd() # using absolute paths
results_path  = os.path.join(path_main,"results")
setup_galaxy  = os.path.join(path_main,"setup_files","galaxy")
setup_cluster = os.path.join(path_main,"setup_files","cluster")

################################################################################
######## Option 1: run the model over the parameters defined in the .dat #######
########           files and define number of iterations in a terminal.  #######
################################################################################

print('╭━━━╮╱╱╭╮╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╭╮')
print('┃╭━╮┃╱╱┃┃╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱┃┃')
print('┃┃╱╰╋━━┫┃╭━━┳╮╭┳╮╱╭╮╱╱╭┳━╮╱╱╱╭━━╮╱╱┃╰━┳━━┳╮╭╮')
print('┃┃╭━┫╭╮┃┃┃╭╮┣╋╋┫┃╱┃┣━━╋┫╭╮┳━━┫╭╮┣━━┫╭╮┃╭╮┣╋╋╯')
print('┃╰┻━┃╭╮┃╰┫╭╮┣╋╋┫╰━╯┣━━┫┃┃┃┣━━┫╭╮┣━━┫╰╯┃╰╯┣╋╋╮')
print('╰━━━┻╯╰┻━┻╯╰┻╯╰┻━╮╭╯╱╱╰┻╯╰╯╱╱╰╯╰╯╱╱╰━━┻━━┻╯╰╯')
print('╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╭━╯┃')
print('╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╰━━╯\n')
print('Welcome!\n\
Note: If you want to commit any changes, please look at cluster_setup,\n\
image_setup or galaxy_setup files before altering the code itself.\n')
it_number = np.int32(input('How many times would you like to run the code?\n\
: '))
if it_number == 0:
	sys.exit('0 is not a valid choice. Program stops execution.')
else:
	pass
print('\nThe code is now running. Please wait.\n')

################################################################################
####### Option 2: run the model for many values as many times as you want ######
####### Note: if you choose option 2, comment lines 39-56. And uncomment  ######
#######       lines below, i.e., 72, 73, 74 and 75.                       ######
################################################################################

#it_number = 1     # here you can manually define the number of iterations
#imf_val   = [0]   # choose the values you want to iterate over (imf type)
#SFE_val   = [0.1] # -||- (star formation efficiency)
#tff_val   = [1.0] # -||- (free-fall time scaling factor)
#If you choose this option, please comment lines 86-88

################################################################################
############## Downloading parameters from the cluster_setup file ##############
################################################################################

cluster_setup = {}
for line in open(os.path.join(path_main,setup_galaxy,"cluster_setup.dat"),"r").readlines():
	cluster_setup[line.split()[0]]=float(line.split()[1])

SFE_val    = [cluster_setup['SFE']]
imf_val    = [int(cluster_setup['imf'])]
tff_val    = [cluster_setup['tff']]
start      = 1           # of the iteration
end        = it_number+1 # end of the iteration
iterat     = np.arange(start,end,1)

################################################################################
################################ Setting paths #################################
################################################################################

# Below we check whether we have properly named folders to store model results.
# Should work with Option 1 and 2.
for s in SFE_val:
	for n in tff_val:
		for j in imf_val:
			gal_name = "Galaxy_tff="+str(n)+"_imf="+str(j)+"_SFE="+str(s)
			gal_path = os.path.join(results_path,gal_name)
			if os.path.isdir(gal_path):
				pass
			else:
				print("Creating new directories for: ",gal_name)
				gal_directory = os.mkdir(gal_path)
				csv_gal       = os.makedirs(os.path.join(gal_path,"csv"))
				fits_gal      = os.mkdir(os.path.join(gal_path,"fits"))
				setups        = os.mkdir(os.path.join(gal_path,"setup"))
				print("Directories successfully created...")
				# Below we ensure that each setup has it's own setup file,
				# so that results are not mixed and each simulation is
				# based on a proper file
				shutil.copy(os.path.join(setup_cluster,"image_setup_change.dat"),\
					 os.path.join(gal_path,"setup"))
				shutil.copy(os.path.join(setup_cluster,"cluster_setup_change.dat"),\
					 os.path.join(gal_path,"setup"))

################################################################################
######## Defining and starting timer to control the time of simulations ########
######## Note: can be commented; t.stop() is the last line of the code. ########
################################################################################
### Can be commented (if you do so, consider commenting it also in the preamble)
t = Timer()
t.start()


#################################
### Predefining plot settings ###
#################################

params={
	'axes.labelsize'	   : 31,
	'axes.linewidth'	   : 1.5,
	'lines.markeredgewidth': 1.5,
	'font.size'			   : 31,
	'legend.fontsize'	   : 31,
	'xtick.labelsize'	   : 31,
	'ytick.labelsize'	   : 31,
	'xtick.major.size'	   : 17,
	'xtick.minor.size'	   : 13,
	'ytick.major.size'	   : 17,
	'ytick.minor.size'	   : 13,
	'savefig.dpi'		   : 300,
	'font.family'		   : 'serif',
	'font.serif'		   : 'Times',
	'text.usetex'		   : True,
	'xtick.direction'      : 'out',     # direction: in, out, or inout
	'xtick.minor.visible'  : True,   	# visibility of minor ticks on x-axis
	'ytick.direction'      : 'out',     # direction: in, out, or inout
	'ytick.minor.visible'  : True,    	# visibility of minor ticks on x-axis
	'xtick.top'			   : True,
	'ytick.right'		   : True,
}
plt.rcParams.update(params)


######################################################
### Creating galaxy spatial and mass distributions ###
######################################################

def spatial_mass_dist(
	NC,                                # Total number of clusters
	mmin,                              # Min. cluster mass
	mmax):                             # Max. cluster mass

    # Read in the 'galaxy_setup.dat' file
	galaxy_setup = {}
	for line in open(os.path.join(setup_galaxy,"galaxy_setup.dat"),"r").readlines():
		galaxy_setup[line.split()[0]]=float(line.split()[1])

    # Read in parameters from the 'galaxy_setup.dat' file
	alpha = galaxy_setup['alpha']      # mass spectrum slope
	A     = galaxy_setup['A']          # scaling parameter
	B     = galaxy_setup['B']          # controls arm sweep and bar size
	N     = galaxy_setup['N']          # controls the amount of winding
	phi   = galaxy_setup['phi']        # phi (angle) coverage (deg)
	hR    = galaxy_setup['hR']         # scale-length
	Rg    = galaxy_setup['Rg']         # galactocentric radius (kpc)

	phi = math.radians(phi)            # convert deg to rad

	### Mass distribution ###
	def mass_spectrum(x,alpha):
		mass_spectrum = x**(-alpha)
		return mass_spectrum

	def mass_dist(
		mmin,
		mmax,
		NC,
		alpha):

		result = []
		while len(result) < NC:
			x = np.random.uniform(np.log10(mmin), np.log10(mmax), size=10*NC)
			y = np.random.uniform(0, 1, size=10*NC)
			result.extend(x[np.where(y < mass_spectrum(x,alpha))])

		md = np.array(result[:NC])
		return 10**md

	MassDistribution = np.asarray(mass_dist(mmin = mmin, mmax = mmax, NC = np.int32(NC), alpha = alpha))
	new_MassDistribution = [[i] for i in MassDistribution]

	### Initial parameters for spatial distribution ###
	### Based on Ringermacher & Mead 2009
	mu = 0                             # centered on function spiral value
	Nd = int(NC)                       # disk cluster number


	### Creating exponential distribution ###
	R    = np.arange(0,Rg,0.0001) #galactocentric radius
	rexp = np.exp(-(R/hR))
	m    = np.random.choice(rexp,int(Nd/(2))) # radially follows an exponential distribution - randomly chosen

	### Generating spiral arm shape given parameters ###
	no  = m*phi #values of phi with designated highest phi value ; deleted: /(max(m))
	X = []
	Y = []

	for i in no:
		r     = A/(math.log(B*math.tan((i)/(2*N))))+1.5 #actual values for spiral arm randomly chosen
		sigma = 1/(2+0.5*phi) #spread from spiral value following normal distribution
		rx    = np.random.normal(mu,sigma,1)
		ry    = np.random.normal(mu,sigma,1)
		x     = r*math.cos(i)+rx
		y     = r*math.sin(i)+ry
		X.append(x)
		Y.append(y)
		x1    = r*math.cos(i+math.pi)+rx
		y1    = r*math.sin(i+math.pi)+ry
		X.append(x1)
		Y.append(y1)

	X = np.array(X).flatten()
	Y = np.array(Y).flatten()
	new_Y = [[-i] for i in Y] ## here we invert y-values to match the galaxy spatial setup
	new_X = [[i] for i in X]
	SpatialArray     = np.append(new_X,new_Y,1)
	SpatialMassArray = np.append(SpatialArray, new_MassDistribution,1)
	SpatialX         = SpatialMassArray[:, 0]
	SpatialY         = SpatialMassArray[:, 1]
	Mass             = SpatialMassArray[:, 2]

	return Mass, SpatialX, SpatialY


galaxy_setup = {}
for line in open(os.path.join(path_main,"setup_files","galaxy","galaxy_setup.dat"),"r").readlines():
	galaxy_setup[line.split()[0]]=float(line.split()[1])
NC    = galaxy_setup['NC']
mmin  = galaxy_setup['mmin']
mmax  = galaxy_setup['mmax']

mass, X, Y = spatial_mass_dist(NC = NC, mmin = mmin, mmax = mmax)

filename = os.path.join(results_path,"galaxycluster_emission.csv")
if os.path.exists(filename):
	os.remove(filename)

DIST_name={}
f=open(os.path.join(setup_galaxy,"image_setup.dat"),"r")
for line in f.readlines():
    DIST_name[line.split()[0]]=line.split()[1]
DIST_name = DIST_name['dist']
f.close()

# SFE, imf and tff are additional definitions. Used to refer to main file.
# Important for the option 2.
SFE    = cluster_setup['SFE']    
imf    = int(cluster_setup['imf'])
tff    = cluster_setup['tff']

Mcm     = cluster_setup['Mcm']
Mcm_gal = np.asarray(mass)	

for z in iterat: # Number of repeated runs of the code with the same parameters (used for creating, e.g., a set of 10 simulations to average the results and get statistically meaningful outcomes)
	for n in tff_val: # Free fall time scalling - the same for all of the clusters, but can be iterated over different values (i.e., produce galaxies with different free fall time scaling applied)
		for j in imf_val: # Form of the initial mass function - the same for all of the clusters, but can be iterated over different values (-||-)
			for s in SFE_val: # Star formation efficiency- the same for all of the clusters, but can be iterated over different values (-||-)

				#################################################################
				gal_vals   = "_tff="+str(n)+"_imf="+str(j)+"_SFE="+str(s)	# string with values describing the galaxy
				iteration  = "_iteration="+str(z)							# string describing the current iteration
				gal_name   = "Galaxy"+gal_vals								# string describing the galaxy
				gal_path   = os.path.join(results_path,gal_name)			# path to results folder of the currently calculated galaxy
				setup_change = os.path.join(gal_path,"setup")				# path to folder with changable setup folder for the currently calculated galaxy
				file_dist  = os.path.join(gal_path,DIST_name)				# path to the distribution.npy (or whatever it's called) file
				csv_path   = os.path.join(gal_path,"csv")					# path to folder with csv files (final result) of the currently calculated galaxy
				fits_path  = os.path.join(gal_path,"fits")					# path to folder with fits files (final result) of the currently calculated galaxy
				fits_file  = "Gal_Template"+gal_vals+iteration+".fits"
				csv_file   = "gal_emission"+gal_vals+iteration+".csv"
				pdf_file   = "Gal_Template"+gal_vals+iteration+".pdf"
				#################################################################

				#################################################################
				if not os.path.exists(os.path.join(csv_path,csv_file)):
					pass
				else:
					os.remove(os.path.join(csv_path,csv_file))
				#--------------------------------------------------------------#
				if not os.path.exists(os.path.join(fits_path,fits_file)):
					pass
				else:
					os.remove(os.path.join(fits_path,fits_file))
				#################################################################
					
				for i in Mcm_gal: # CLUSTER LAYER THAT CHANGES - Mcm_gal contains masses of 1e4 GMCs in 1 galaxy

					#################################################################								
					Mcm_value  = "_Mcm="+str(i)										# string with the currently used molecular cloud mass												
					npy_file   = "distributions"+Mcm_value+gal_vals+iteration+".npy"
					#################################################################

					f = open(os.path.join(setup_galaxy,"cluster_setup.dat"),'r') 			    # Here we open the template setup file
					fout = open(os.path.join(setup_change,"cluster_setup_change.dat"), "w+")	# and here we open file which copies the unchangable values from the 'f' file, can replace molecular cloud mass
					
					for line in f:
						line = line.replace(str(Mcm), str(i)) # take i instead of current molecular cloud mass
						fout.write(line)

						for line in f:
							line = line.replace(str(SFE), str(s)) # takes(specified at the beginning of the code due to iteration possiblity) instead of SFE from the file
							fout.write(line)

							for line in f: # take j(specified at the beginning of the code due to iteration possiblity) instead of IMF from the file
								line = line.replace(str(imf),str(j),1)
								fout.write(line)

								for line in f: # take n(specified at the beginning of the code due to iteration possiblity) instead of tff from the file
									line = line.replace(str(tff),str(n))
									fout.write(line)

					fout.close() #close the changable file
					f.close()    #close the original file

					outputfile   = os.path.join(csv_path,csv_file) # "_date="+timestr+ ; add this in case you want a time stamp as a part of the outputfile
					newname	     = os.path.join(gal_path,npy_file)

					c_change     = os.path.join(setup_change,"cluster_setup_change.dat")
					i_change     = os.path.join(setup_change,"image_setup_change.dat")

					distribution = cd.Mod_distribution(FILE_cluster = c_change)
					distribution.calc(output = 0, FILE_dist = file_dist) # here we create 'distribution.npy' file
					
					os.rename(file_dist,newname) # here we change 'distribution.npy' name to one that describes the galaxy
					
					g    = open(os.path.join(setup_galaxy,"image_setup.dat"))

					gout = open(i_change, "wt")
					for line in g:
						gout.write(line.replace(DIST_name, npy_file)) 
					g.close() 
					gout.close()
					
					template = ce.Mod_Template()
					template.main(output = 0,FILE = outputfile, SETUP_image = i_change, PATH_dist = gal_path)
					
					try:
						os.remove(newname)
					except OSError:
						pass

					# here all of the calculations for 1 cluster end, 1 line is appended in the output file and the loop goes over the next

				im = []
				mass = []
				N = []

				with open(outputfile, 'r', newline='') as file: # added newline='' because Windows
					reader = csv.reader(file, delimiter='\t')
					for row in reader:
						im.append(float(row[0]))
						mass.append(float(row[1]))
						N.append(float(row[2]))

				mass = [[i] for i in mass]
				ims  = [[i] for i in im]
				N	= [[i] for i in N]

				comb  = np.append(ims,mass,1)
				comb  = np.append(comb,N,1)

				# Take spatial distribution and place emitting clusters on a galactic grid that will be a part of an intensity map
				galaxy_setup = {}
				for line in open(os.path.join(setup_galaxy,"galaxy_setup.dat"),"r").readlines():
					galaxy_setup[line.split()[0]]=float(line.split()[1])

				Rg = galaxy_setup['Rg']
				gamma = galaxy_setup['gamma']# slope for mass-size relation

				config = {}
				f = open(os.path.join(setup_change,"image_setup_change.dat"),"r")
				for line in f.readlines():
					config[line.split()[0]]=line.split()[1]

				# Parameters relating to new image
				dist	   = float(config['bob'])   # distance to cluster in pc
				pixel_size = float(config['psize']) # pixel size in arcsec
				resolution = float(config['beam'])  # resolution of new image
				dim_pix	   = int(config['dim'])	    # image size in pixels
				npix_beam  = 2.*np.pi*(resolution/2./(2.*np.log(2.))**0.5)**2 / pixel_size**2   # number of pixels per beam

				if np.max(X) >= abs(np.min(X)):
					max_x = np.max(X)
				else:
					max_x = abs(np.min(X))
				if np.max(Y) >= abs(np.min(Y)):
					max_y = np.max(Y)
				else:
					max_y = abs(np.min(Y))

				dims = (dim_pix ,dim_pix)                     # total grid dimensions
				half_im = dim_pix / 2
				safety_margin = 20                            # ensures that galaxy doesn't exceed the image and there are few pixels free from emission on each side
				Galaxyarray = np.zeros(dims)
				conv_ = 206264.806247                         # arcsec-to-pc conversion
				pix_  = (pixel_size/conv_)*dist               # pixel size in data coorindates; here in pc

				Galaxyarray = np.zeros(dims)
				for i in range(0,len(X)):
					R = ((1/155)*comb[i][1])**(1/gamma) # mass-size relation (M =~ R^2); Lada & Dame 2020

					if 2*R > pix_:           # for clouds that are bigger than 1 pixel size (see pixel_size_template_galaxy)
						dim  = int(2*R/pix_) # how many pixels corresponds to the size of a cloud
						d    = comb[i][0]/(dim**2)                 # flux/area (area = in pixels; so 1 pixel = 1 unit, 2 pixels gives as 2x2 etc.)
						data = np.zeros((dim,dim))                 # define array that will correspond to fluxes per pixel for a specific cloud
						data.fill(d)                               # fill the data array with corresponding fluxes

					else:
						d    = comb[i][0]                          # take intensity
						data = np.zeros((3,3))
						data[1,1] = d

					x = X[i]*((half_im-safety_margin)/max_x)
					y = Y[i]*((half_im-safety_margin)/max_y)
					Galaxyarray[int((x+(dims[0]-len(data))/2)):int((x+(dims[0]+len(data))/2)),int((y+(dims[1]-len(data))/2)):int((y+(dims[1]+len(data))/2))]+=data

				beam   = Gaussian2DKernel(resolution/pixel_size/(2.*(2.*np.log(2.))**0.5))
				im_obs = convolve(Galaxyarray, beam, boundary='extend')/npix_beam

				#im_obs[im_obs==0] = 1e-100
				rnge = (pix_*dim_pix/2.)/1000.        # 1. kpc to arcmin: Comment this one
				# Comment the the lines below until 'hdu.writeto' if you're not interested in producing a fits file
				header = fits.Header()
				header['BMAJ']    = resolution / 3600.
				header['BMIN']    = resolution / 3600.
				header['BPA']     = 0.0
				header['BTYPE']   = 'Intensity'
				header['BUNIT']   = 'JY KM/S /BEAM '
				header['EQUINOX'] = 2.000000000000E+03
				header['CTYPE1']  = 'RA---SIN'
				header['CRVAL1']  = 0.0
				header['CDELT1']  =  pixel_size/3600.
				header['CRPIX1']  =  half_im
				header['CUNIT1']  = 'deg     '
				header['CTYPE2']  = 'DEC--SIN'
				header['CRVAL2']  = 0.0
				header['CDELT2']  =  pixel_size/3600.
				header['CRPIX2']  =  half_im
				header['CUNIT2']  = 'deg     '
				header['RESTFRQ'] = 9.879267000000E+11
				header['SPECSYS'] = 'LSRK    '
				hdu = fits.PrimaryHDU(im_obs, header=header)
				hdu.writeto(os.path.join(fits_path,fits_file), overwrite = True) #"_date="+timestr+

				print ("Peak intensity in image %s is %6.5f Jy km/s/beam" %(str(z),im_obs.max()))

				#'''
				### Image plotting ###
				fig, ax = plt.subplots(figsize=[13,10])
				my_cmap = copy.copy(matplotlib.cm.get_cmap('cividis)) # copy the default cmap
				my_cmap.set_bad([0,0,0])

				i_map = ax.imshow(
				im_obs,
				#interpolation='nearest',
				cmap = my_cmap,
				aspect = 'equal',
				vmin=0, vmax=np.amax(im_obs),   #norm = LogNorm()
				extent=(rnge,-rnge,-rnge,rnge)
				)

				cbar = plt.colorbar(i_map)
				cbar.set_label('Jy km s$^{-1}$ beam$^{-1}$',labelpad=15)
				#ax.contour(np.arange(1,1402,1),np.arange(1,1402,1),im_obs,origin="lower",colors='yellow',levels=[0.03],linewidths=0.2)
				#ax.set_xlabel('Offset (arcmin)')
				#ax.set_ylabel('Offset (arcmin)')
				ax.set_xlabel('Offset (kpc)')
				ax.set_ylabel('Offset (kpc)')

				#######################################################################################
				########## Adding an inset with a beam size and a close up on the chosen region #######
				#######################################################################################
				#axins = ax.inset_axes([0.075, 0.072, 0.14, 0.14]) # [x0, y0, width, height] = lower-left corner of inset axes, and its width and height
				#axins.imshow(
				#im_obs,
				#interpolation='nearest',
				#cmap = my_cmap,
				#norm = LogNorm(vmin=1e-6, vmax=np.amax(im_obs)),
				#extent=(12,-12,-12,12)
				#)

				#x1, x2, y1, y2 = 5, 3, -2.5, -2           # Choosing a region to make a close up from
				#axins.set_xlim(x1, x2)                      # "Cut" the inset image [x-axis] = show only the region you are interested in
				#axins.set_ylim(y1, y2)                      # "Cut" the inset image [y-axis] = show only the region you are interested in
				#axins.yaxis.get_major_locator().set_params(nbins=2)
				#axins.xaxis.get_major_locator().set_params(nbins=2)
				#axins.tick_params(axis='both', colors='white',length=12,labelsize=20)
				#axins.set_color(axis='both', color='white')
				#axins.spines['bottom'].set_color('white')
				#axins.spines['top'].set_color('white')
				#axins.spines['right'].set_color('white')
				#axins.spines['left'].set_color('white') #--end od uncommenting
				#ax.indicate_inset_zoom(axins, edgecolor="white")
				#plt.xticks(visible=False)
				#plt.yticks(visible=False)

				############################## Add beam to your plot ##################################
				#beam_ellipse  = (resolution/conv_)*dist/1000. # beam size in data coorindates; here in kpc
				# (1) First we create a class of an anchored ellipse
				#class AnchoredEllipse(AnchoredOffsetbox):
				#	def __init__(self, transform, width, height, angle, loc,
				#				pad=0.1, borderpad=0.1, prop=None, frameon=False):
				#		self._box = AuxTransformBox(transform)
				#		self.ellipse = Ellipse((0, 0), width, height, angle,fill=False,color='white',hatch='//////')
				#		self._box.add_artist(self.ellipse)
				#		super().__init__(loc, pad=pad, borderpad=borderpad,
				#						child=self._box, prop=prop, frameon=frameon)
				# (2) Define your ellipse
				#def draw_ellipse(ax):
				#	ae = AnchoredEllipse(axins.transData, width=beam_ellipse, height=beam_ellipse, angle=0.,
				#						loc='lower left', pad=0.5, borderpad=0.2,
				#						frameon=False)
				#	ae.patch.set_facecolor('none')
				#	ae.patch.set_edgecolor('none')
				#	axins.add_artist(ae)
				# (3) Draw it (comment the line below if you don't want to add your beam)
				#draw_ellipse(axins)

				plt.savefig(os.path.join(results_path,pdf_file),bbox_inches='tight') #"_date="+timestr+
				#'''

t.stop() # Comment this if you commented the timer at the beginning of the code!
