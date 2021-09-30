# galaxy-in-a-box
Galaxy-in-a-box model

Statistical model of sub-millimeter emission from embedded star forming clusters in galaxies. The paper describing the model is currently in preparation (Dutkowska, Kristensen & Bergin 2021). The model is written in Python 3 and uses following packages:
- astropy
- matplotlib
- mpl_toolkits
- numpy
- scipy
- sympy
- pylab

The galaxy-in-a-box model consists of three scripts stored in **model**. The first (cluster_distribution) generates the cluster based on the number of stars, input initial mass function, spatial distribution and age distribution. The second (cluster_emission) takes an input file from the [Water Emission Database](https://katarzynadutkowska.github.io/WED/) with a set of observational data, determines the mass-intensity correlation and generates outflow emission for all low-and intermediate-mass Class 0 and I sources and high-mass protostars (which here follow the same age distribution as low- and intermediate-mass protostars). The output is stored as a CSV file. The third (galaxy_emission) create a spatial and mass distribution of giant molecular clouds along galactic arms, and then using two cluster scripts to simulate emission from clusters forming from these giant molecular clouds, and the output is stored in a form of a FITS image where the flux density is determined by the desired resolution, pixel scale and galaxy distance, and in a form of a CSV file with mass, flux and numbers or stars of each galactic cluster.

Setup files where the most important global parameters can be changed are stored in **/model/setup_files**. In case of running only the **cluster part** of the model (i.e., scripts: cluster_distribution and cluster_emission), please change the values within files ending **with** 'change'. In case of running the **galaxy part** (galaxy_emission) please upadate the files **without 'change'** at the end.
