# Galaxy-in-a-box

Statistical model of sub-millimeter emission from embedded star forming clusters in galaxies. The paper describing the model is already submitted (Dutkowska 
& Kristensen 2022). The model is written in Python 3 and uses following packages that may require installation:
- astropy
- matplotlib
- mpl_toolkits
- numpy
- scipy
- sympy
- pylab

The galaxy-in-a-box model consists of three scripts stored in the **model** folder. The first (cluster_distribution) generates the star-forming cluster based on the number of stars, input initial mass function, spatial distribution and age distribution. The second (cluster_emission) takes an input data from the [Water Emission Database](https://katarzynadutkowska.github.io/WED/), determines the mass-intensity correlation and generates outflow emission for all low- to intermediate-mass Class 0 and I sources and high-mass protostars (here, they follow the same age distribution as the low- and intermediate-mass protostars). The output is stored as a CSV file. Both of the cluster scripts are based on the [**cluster-in-a-box** model by Kristensen & Bergin](https://github.com/egstrom/cluster-in-a-box). The third script (galaxy_emission) creates a spatial and mass distribution of giant molecular clouds along galactic arms, and then use two cluster scripts to simulate emission from clusters forming from these giant molecular clouds, and the output is stored in a form of a FITS image where the flux density is determined by the desired resolution, pixel scale and galaxy distance, and in a form of a CSV file with mass, flux and numbers or stars of each galactic cluster. There is a possibility to save a publication-ready integrated intensity map of the galaxy in a PDF format.

Setup files where the most important global parameters can be changed are stored in **/model/setup_files**. In case of running only the **cluster part** of the model (i.e., scripts: cluster_distribution and cluster_emission), please change the values within files ending **with 'change'**. In case of running the **galaxy part** (galaxy_emission) please upadate the files **without 'change'** at the end.
