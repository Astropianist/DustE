# DustE
Calculate dust attenuation curves for various configurations of physical properties of galaxies like stellar mass, star formation rates, and metallicities. This code is designed with theorists in mind: by supplying lists of non-dust properties as mentioned above, they can get state-of-the-art estimates for the dust properties of their simulated galaxies. Of course, observers can also compare their findings to our models, which will be a useful validation process. 

## Background
We created hierarchical/population Bayesian models for dust attenuation, which are described in Nagaraj+22a (submitted). Using Prospector spectral energy distribution (SED) fitting posterior samples for nearly 30,000 galaxies in a mass-complete 3D-HST (http://3dhst.research.yale.edu/Home.html) sample ([Leja et al. 2017](https://ui.adsabs.harvard.edu/abs/2017ApJ...837..170L/abstract), [2019](https://ui.adsabs.harvard.edu/abs/2019ApJ...877..140L/abstract), [2020](https://ui.adsabs.harvard.edu/abs/2020ApJ...893..111L/abstract)), we fit a linear interpolation function in five dimensions in order to determine how the dust attenuation curve varies over physical properties like stellar mass and star formation rate. As a result, we have created four models that are available for the user. We have both two-component (diffuse and birth cloud) and single-component (effective) dust attenuation model options. In the case of the two-component models, the birth cloud dust optical depth is modeled as a 1-D interpolation function of diffuse dust optical depth, which we provide in the form of a convenience function get_d1 (method of the DustAttnCalc class, see below). For diffuse dust and effective dust, we have a model that predicts both the dust optical depth and slope of the attenuation curve as well as a model that requires dust optical depth as an input to calculate the slope of the attenuation curve. The model of choice can be chosen through Boolean options for the DustAttnCalc class instance

The two-component model, inspired by the great success of the [Charlot & Fall (2000)](https://ui.adsabs.harvard.edu/abs/2000ApJ...539..718C/abstract) model in describing real galaxies, is used by Prospector in SED fitting. The model considers that the light from stars passes through either one or two dust screens in front of the galaxy. Light from stars under 10 Myr old pass through both screens (diffuse and birth cloud dust) whereas light from all other stars pass through only the diffuse dust screen. The birth cloud dust attenuation curve is treated as a simple inverse law with the normalization optical depth at 550 nm. The diffuse dust attenuation curve is parameterized in the form used by [Noll et al. (2009)](https://ui.adsabs.harvard.edu/abs/2009A%26A...507.1793N/abstract) and [Kriek & Conroy (2013)](https://ui.adsabs.harvard.edu/abs/2013ApJ...775L..16K/abstract), which is a flexible generalization of the [Calzetti et al. (2000)](https://ui.adsabs.harvard.edu/abs/2000ApJ...533..682C/abstract) curve. We use this same parameterization for the effective dust curve as well. For the effective dust attenuation case, there is simply one dust screen that affects all stars regardless of age.
## Installation
To install the package, open terminal in the directory where you would like the code to be placed and type
        
        git clone https://github.com/Astropianist/DustE.git

### Dependencies
A few packages that are available through pip and/or Anaconda are required to run the code.

* NumPy (tested on 1.20.0)
* SciPy (tested on 1.6.0)
* ArViz (tested on 0.11.2)
* Matplotlib (tested on 3.4.2)
* Seaborn (tested on 0.11.1)
* Astropy (tested on 4.2)
* [Sedpy](https://github.com/bd-j/sedpy)

## Using the code
The main functionality of the code comes from the DustAttnCalc class of DustAttnCalc.py. It can be used to calculate and plot dust attenuation curves depending on values of the independent parameters (log stellar mass, log star formation rate, log stellar metallicity, redshift, axis ratio, or even the dust optical depth). Here is an example, with the code situated in DustE/.

        from DustAttnCalc import *
        import numpy as np
        
        ngal = 100 # Number of galaxies to use
        # Set stellar mass, star formation rate, and metallicity within the bounds suggested in the code
        logM = np.random.uniform(8.74,11.30,ngal)
        sfr = np.random.uniform(-2.06,2.11,ngal)
        logZ = np.random.uniform(-1.70,0.18,ngal)
        dust_attn = DustAttnCalc(logM=logM, sfr=sfr, logZ=logZ, bv=1, eff=0) # Two-component bivariate dust model (fitting both optical depth and slope) 
        dac, dac1 = dust_attn.calcDust(plot_tau=True, max_num_plot=5) # This line will calculate the diffuse and birth cloud dust attenuation curves for all 100 galaxies created earlier. It will also produce 5 dust attenuation plots showing both diffuse and birth cloud dust (from the argument plot_tau1=True). The 5 points will be selected randomly from the 100 galaxies created earlier.
        
As mentioned in the code comments, the code snippet above will measure the dust attenuation curves for 100 galaxies and create 5 plots of diffuse and birth cloud dust, with points taken randomly from the 100 galaxies provided. By default, the images will be stored in a directory called DustAttnCurves and named DustAttnCurve_bv_1_eff_0_0i, with i ranging from 0 to 4. Here is an example of the type of image that would be produced from the code above.

<p align="center">
  <img src="DustAttnCurve_bv_1_eff_0_01.png" width="650"/>
</p>
