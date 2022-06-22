# Galaxy-in-a-box

<p style="text-align:center;">
     <img src="https://katarzynadutkowska.github.io/WED/Images/gal-in-a-box-shadow.jpeg"
     alt="GIAB icon"
     width="30%" />
</p>

Statistical model of sub-millimeter emission from embedded star forming clusters in galaxies. The paper describing the model is in press ([Dutkowska 
& Kristensen 2022](https://ui.adsabs.harvard.edu/abs/2022arXiv220601753D/abstract)). The model is written in Python 3 and uses following packages that may require installation:
- astropy
- matplotlib
- mpl_toolkits
- numpy
- scipy
- sympy
- pylab

The galaxy-in-a-box model consists of three scripts stored in the **model** folder. The first (cluster_distribution) generates the star-forming cluster based on the number of stars, input initial mass function, spatial distribution and age distribution. The second (cluster_emission) takes an input data from the [Water Emission Database](https://katarzynadutkowska.github.io/WED/), determines the mass-intensity correlation and generates outflow emission for all low- to intermediate-mass Class 0 and I sources and high-mass protostars (here, they follow the same age distribution as the low- and intermediate-mass protostars). The output is stored as a CSV file. Both of the cluster scripts are based on the [**cluster-in-a-box** model by Kristensen & Bergin](https://github.com/egstrom/cluster-in-a-box). The third script (galaxy_emission) creates a spatial and mass distribution of giant molecular clouds along galactic arms, and then use two cluster scripts to simulate emission from clusters forming from these giant molecular clouds, and the output is stored in a form of a FITS image where the flux density is determined by the desired resolution, pixel scale and galaxy distance, and in a form of a CSV file with mass, flux and numbers or stars of each galactic cluster. There is a possibility to save a publication-ready integrated intensity map of the galaxy in a PDF format.

Instructions to run the model:
1. Run the model from the **/model/.** directory, where all of the codes are saved. 
2. In order to run only the **cluster part**:
   * Edit the setup files in **/model/setup_files/cluster** directory.
   * First you need to run the cluster_distribution files, and only then cluster_emission,
   * All results will be saved in **/model/results**.
3. In order to run the **galaxy part**:
   * Edit the setup files in **/model/setup_files/galaxy** directory.[^1]
   * Run the galaxy_emission file. The program will ask about the number of iterations (i.e., how many times to run the model with the desired setup; for example to get enough results to make a statistically meaningful analysis of the predicted emission).
   * All of the files (CSV, FITS and a saved copy of setup files) will be saved in a corresponding folder in **/model/results/**, e.g., Galaxy_tff=1.0_imf=0_SFE=0.1.

[^1]: Currently, it is possible to run the model in two ways, referred to as **Option 1** and **Option 2** in galaxy_emission.py. **Option 1** allows you to run the model from the terminal where you define the number of iterations over the same input parameters (which can be changed through the setup files). However, it is possible to run the model over different input parameters (based on the Cartesian product from the parameter lists) many times (i.e., with many iterations) when choosing **Option 2**. **Option 1** is the default one. The change of the active option is possible through commenting and uncommenting parts of the code (galaxy_emission.py). The exact instructions on which lines require commenting or uncommenting are stated in the code.




***Known issues***
Windows users may experience an error while running the galaxy_emission.py file. The error results from adding an empty line in the output CSV file, which happens through the Windows system. It will soon be fixed.
