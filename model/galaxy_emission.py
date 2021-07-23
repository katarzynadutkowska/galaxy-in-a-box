# Galaxy-in-a-box authors: Katarzyna M. Dutkowska, Lars E. Kristensen,
# Markus Rasmussen, Thomas Jones
# Copenhagen University, NBI & StarPlan, Copenhagen, Denmark
# 2021

import numpy as np
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
import Galaxy_clusters as cd
import Galaxycluster_emission as ce
import time
timestr = time.strftime("%Y%m%d-%H%M%S") # Here we save the time of running the
                                         # script; used later to save files
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
Note: If you want to change something, please look at cluster_setup,\n\
image_setup or galaxy_setup files before you change something in the code.\n')
it_number = np.int32(input('How many times would you like to run the code?\n\
: '))
if it_number == 0:
	sys.exit('0 is not a valid choice. Program stops execution.')
else:
	pass
print('The code is now running. Please wait.')
################################################################################

### Calling the imported timer function to control the time of calculations
#   Can be commented (if you do so, consider commenting it also in the preamble)
t = Timer()
t.start() # t.stop() is the last line of the code

#################################
### Predefining plot settings ###
#################################

params={
	'axes.labelsize'	   : 29,
	'axes.linewidth'	   : 1.5,
	'lines.markeredgewidth': 1.5,
	'font.size'			   : 29,
	'legend.fontsize'	   : 24,
	'xtick.labelsize'	   : 29,
	'ytick.labelsize'	   : 29,
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
	for line in open("./setup_files/galaxy_setup.dat","r").readlines():
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
	MassDist  = []
	massrange = np.arange(mmin,mmax,1) # create an array with masses in range [mmin, mmax) with a step = 1
	MassDist  = massrange**(alpha)     # create a mass distribution based on alpha
	MassDistribution = (MassDist/max(MassDist))*mmax

	mass = []
	while len(mass) < (NC):
		MassDist = np.random.choice(MassDistribution,size = 1)
		if MassDist > mmin:
			mass.append(MassDist)
	new_MassDistribution = [[i[0]] for i in mass]


	### Initial parameters for spatial distribution ###
	# Ringermacher & Mead 2009
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
	Mass             = (SpatialMassArray[:, 2]).flatten()

	return Mass, SpatialX, SpatialY


galaxy_setup = {}
for line in open("./setup_files/galaxy_setup.dat","r").readlines():
	galaxy_setup[line.split()[0]]=float(line.split()[1])
NC    = galaxy_setup['NC']
mmin  = galaxy_setup['mmin']
mmax  = galaxy_setup['mmax']

mass, X, Y = spatial_mass_dist(NC = NC, mmin = mmin, mmax = mmax)

filename = './results/galaxycluster_emission.csv'
if os.path.exists(filename):
	os.remove(filename)

cluster_setup = {}
for line in open("./setup_files/cluster_setup.dat","r").readlines():
	cluster_setup[line.split()[0]]=float(line.split()[1])

# For iterations uncomment these lines (and comment the ones that are currently active, besides 'Mcm')
# SFE = [,] # Here provide all of the values which you want to run the model over
# IMF = [,] # Here provide all of the values which you want to run the model over
# tff = [,] # Here provide all of the values which you want to run the model over

SFE     = [cluster_setup['SFE']]
IMF     = [cluster_setup['imf']]
tff     = [cluster_setup['tff']]
start   = 1
end     = it_number+1
iterat  = np.arange(start,end,1)

Mcm     = cluster_setup['Mcm']
Mcm_gal = np.asarray(mass)

for z in iterat:
	for n in tff:
		for j in IMF:
			for s in SFE:
				for i in Mcm_gal:
					f = open("./setup_files/cluster_setup.dat")
					fout = open("./setup_files/cluster_setup_change.dat", "wt")
					for line in f:
						fout.write(line.replace(str(Mcm), str(i)))
						for line in f:
							fout.write(line.replace(str(SFE), str(s)))
							for line in f:
								fout.write(line.replace(str(IMF),str(j)))
								for line in f:
									fout.write(line.replace(str(tff),str(n)))
					f.close()
					fout.close()
					outputfile = "./results/gal_emission"+"_tff="+str(n)+"_imf="+str(j)+"_SFE="+str(s)+"_iteration="+str(z)+"_date="+timestr+".csv"
					newname    = "./results/distributions_Mcm="+str(i)+"_tff="+str(n)+"_imf="+str(j)+"_SFE="+str(s)+"_iteration="+str(z)+".npy"
					distribution = cd.Mod_distribution()
					distribution.calc(output = 0)
					os.rename("./results/distribution.npy",newname)
					g = open("./setup_files/image_setup.dat")
					gout = open("./setup_files/image_setup_change.dat", "wt")
					for line in g:
						gout.write(line.replace('./results/distribution.npy', newname))
					g.close()
					gout.close()
					template = ce.Mod_Template()
					template.main(output = 0,FILE = outputfile)
					try:
						os.remove(newname)
					except OSError:
						pass

				im = []
				mass = []
				N = []

				with open(outputfile, 'r') as file:
					reader = csv.reader(file, delimiter='\t')
					for row in reader:
						im.append(float(row[0]))
						mass.append(float(row[1]))
						N.append(float(row[2]))

				mass = [[i] for i in mass]
				ims  = [[i] for i in im]
				N    = [[i] for i in N]

				comb  = np.append(ims,mass,1)
				comb  = np.append(comb,N,1)

				# Take spatial distribution and place emitting clusters on a galactic grid that will be a part of an intensity map
				galaxy_setup = {}
				for line in open("./setup_files/galaxy_setup.dat","r").readlines():
					galaxy_setup[line.split()[0]]=float(line.split()[1])

				Rg = galaxy_setup['Rg']
				gamma = galaxy_setup['gamma']# slope for mass-size relation

				config = {}
				f = open('./setup_files/image_setup_change.dat','r')
				for line in f.readlines():
					config[line.split()[0]]=line.split()[1]

				# Parameters relating to new image
				dist       = float(config['bob'])   # distance to cluster in pc
				pixel_size = float(config['psize']) # pixel size in arcsec
				resolution = float(config['beam'])  # resolution of new image
				dim_pix    = int(config['dim'])     # image size in pixels
				npix_beam  = 2.*np.pi*(resolution/2./(2.*np.log(2.))**0.5)**2 / pixel_size**2   # number of pixels per beam

				if np.max(X) >= abs(np.min(X)):
					max_x = np.max(X)
				else:
					max_x = abs(np.min(X))
				if np.max(Y) >= abs(np.min(Y)):
					max_y = np.max(Y)
				else:
					max_y = abs(np.min(Y))

				def noisy(image): # generate and add gaussian noise to emission map; not in use
					row,col = image.shape
					mean = 0
					var = 0.001
					sigma = var**0.5
					gauss = np.random.normal(mean,sigma,(row,col))
					gauss = gauss.reshape(row,col)
					noisy = image + gauss
					return noisy

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

				#noisy_image = noisy(Galaxyarray)
				beam = Gaussian2DKernel(resolution/pixel_size/(2.*(2.*np.log(2.))**0.5))
				im_obs = convolve(Galaxyarray, beam, boundary='extend')/npix_beam

				#im_obs[im_obs==0] = 1e-100
				rnge = (pix_*dim_pix/2.)/1000.        # 1. arcmin -> kpc: Comment this one
				# Comment the the lines below until 'hdu.writeto' if you're not interested in producing a fits file
				header = fits.Header()
				header['BMAJ'] = resolution / 3600.
				header['BMIN'] = resolution / 3600.
				header['BPA'] = 0.0
				header['BTYPE'] = 'Intensity'
				header['BUNIT'] = 'JY KM/S /BEAM '
				header['EQUINOX'] = 2.000000000000E+03
				header['CTYPE1'] = 'RA---SIN'
				header['CRVAL1'] = 0.0
				header['CDELT1'] =  pixel_size/3600.
				header['CRPIX1'] =  half_im
				header['CUNIT1'] = 'deg     '
				header['CTYPE2'] = 'DEC--SIN'
				header['CRVAL2'] = 0.0
				header['CDELT2'] =  pixel_size/3600.
				header['CRPIX2'] =  half_im
				header['CUNIT2'] = 'deg     '
				header['RESTFRQ'] =   9.879267000000E+11
				header['SPECSYS'] = 'LSRK    '
				hdu = fits.PrimaryHDU(im_obs, header=header)
				hdu.writeto("./results/Gal_Template"+"_tff="+str(n)+"_imf="+str(j)+"_SFE="+str(s)+"_iteration="+str(z)+"_date="+timestr+".fits", overwrite = True)

				print ("Peak intensity in image %s is %6.5f Jy km/s/beam" %(str(z),im_obs.max()))

				### Image plotting ###
				fig, ax = plt.subplots(figsize=[15,12])
				my_cmap = copy.copy(matplotlib.cm.get_cmap('pink')) # copy the default cmap
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
				cbar.set_label('Jy km s$^{-1}$ beam$^{-1}$')
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
				rad_to_arcsec = 206264.806247                         # radian-to-arcsec conversion
				beam_ellipse  = (resolution/rad_to_arcsec)*dist/1000. # beam size in data coorindates; here in kpc
				#beam_ellipse  = 0.2368 # M51 'at z=3' with beam of 0.03''
				# (1) First we create a class of an anchored ellipse
				class AnchoredEllipse(AnchoredOffsetbox):
					def __init__(self, transform, width, height, angle, loc,
								pad=0.1, borderpad=0.1, prop=None, frameon=False):
						self._box = AuxTransformBox(transform)
						self.ellipse = Ellipse((0, 0), width, height, angle,fill=False,color='white',hatch='//////')
						self._box.add_artist(self.ellipse)
						super().__init__(loc, pad=pad, borderpad=borderpad,
										child=self._box, prop=prop, frameon=frameon)
				# (2) Define your ellipse
				def draw_ellipse(ax):
					ae = AnchoredEllipse(axins.transData, width=beam_ellipse, height=beam_ellipse, angle=0.,
										loc='lower left', pad=0.5, borderpad=0.2,
										frameon=False)
					ae.patch.set_facecolor('none')
					ae.patch.set_edgecolor('none')
					axins.add_artist(ae)
				# (3) Draw it (comment the line below if you don't want to add your beam)
				#draw_ellipse(axins)

				plt.savefig("./results/Gal_Template"+"_tff="+str(n)+"_imf="+str(j)+"_SFE="+str(s)+"_iteration="+str(z)+"_date="+timestr+".pdf",bbox_inches='tight')

t.stop() # Comment this if you commented the timer at the beginning of the code!
